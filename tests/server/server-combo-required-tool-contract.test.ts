import { afterEach, beforeEach, describe, expect, setDefaultTimeout, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ManagementRequest as Request } from "../helpers/management-auth";
import { comboProviderFactory } from "../helpers/combo-provider";
import { installIsolatedCodexHome, type IsolatedCodexHome } from "../helpers/isolated-codex-home";
import { acquireOwnedSpendHome } from "../helpers/owned-spend-home";
import { removeTreeWithRetry } from "../helpers/remove-tree";
import { clearComboSelectionState, clearComboTargetCooldowns } from "../../src/combos";
import { clearComboRecallForTests } from "../../src/server/responses/combo-session-recall";
import { clearKeyCooldowns } from "../../src/providers/key-failover";
import { clearCodexUpstreamHealth } from "../../src/codex/routing";
import { clearRequestLogsForTests, type RequestLogContext } from "../../src/server/request-log";
import {
  clearResponseStateForTests,
  flushResponseState,
  responseStatePersistPendingForTests,
} from "../../src/responses/state";
import type { OcxConfig } from "../../src/types";

/**
 * #7456: a Responses combo child that answers a `tool_choice`-required request with a buffered
 * `status: "completed"` body carrying zero call items is a false success. The combo used to
 * accept any 200 before commit, so the client received `completed` with nothing to execute and
 * its agent loop ended silently — the hop machinery was only reachable from a non-OK response.
 *
 * The scope is exactly the safe half of the issue's acceptance criteria: buffered non-streaming
 * successful children, requests that explicitly require a tool, judged before any commit.
 * Streaming children keep their conservative preflight rules; the last tests here pin that no
 * judgment reaches past them.
 */

// Loopback upstreams plus combo failover exceed the 5s default under full-suite load on Windows.
setDefaultTimeout(30_000);

let testDir = "";
let previousHome: string | undefined;
let isolatedCodexHome: IsolatedCodexHome | null = null;
const servers: Array<ReturnType<typeof Bun.serve>> = [];
const provider = comboProviderFactory(() => undefined);
let releaseSpendHome: (() => void) | undefined;

beforeEach(() => {
  previousHome = process.env.OPENCODEX_HOME;
  isolatedCodexHome = installIsolatedCodexHome("ocx-combo-required-tool-codex-");
  testDir = mkdtempSync(join(tmpdir(), "ocx-combo-required-tool-"));
  process.env.OPENCODEX_HOME = testDir;
  // Direct handler dispatches need the writer lease that startServer normally holds.
  releaseSpendHome = acquireOwnedSpendHome();
  clearComboSelectionState();
  clearComboRecallForTests();
  clearComboTargetCooldowns();
  clearKeyCooldowns();
  clearCodexUpstreamHealth();
  clearRequestLogsForTests();
  clearResponseStateForTests();
});

afterEach(async () => {
  releaseSpendHome?.();
  releaseSpendHome = undefined;
  let responseStatePending = true;
  try {
    for (const server of servers.splice(0)) await server.stop(true);
    await flushResponseState();
    responseStatePending = responseStatePersistPendingForTests();
  } finally {
    clearResponseStateForTests();
    if (previousHome === undefined) delete process.env.OPENCODEX_HOME;
    else process.env.OPENCODEX_HOME = previousHome;
    isolatedCodexHome?.restore();
    isolatedCodexHome = null;
    if (testDir) removeTreeWithRetry(testDir);
    clearComboSelectionState();
    clearComboRecallForTests();
    clearComboTargetCooldowns();
    clearKeyCooldowns();
    clearCodexUpstreamHealth();
    clearRequestLogsForTests();
  }
  expect(responseStatePending).toBe(false);
});

/** Loopback upstream whose lifetime the afterEach owns. */
function serve(handler: (request: Request) => Response | Promise<Response>) {
  const server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: handler });
  servers.push(server);
  return server;
}

function baseUrl(server: ReturnType<typeof Bun.serve>): string {
  return `${server.url.toString().replace(/\/$/, "")}/v1`;
}

function comboConfig(providers: OcxConfig["providers"]): OcxConfig {
  return {
    port: 0,
    defaultProvider: Object.keys(providers)[0]!,
    providers,
    combos: {
      free: {
        strategy: "failover",
        targets: Object.keys(providers).map((name, index) => ({ provider: name, model: `m${index + 1}` })),
      },
    },
  };
}

/** A completed buffered Responses body whose output is only a text message. */
function completedNoCall(model: string, text: string): Record<string, unknown> {
  return {
    id: `resp-${model}`,
    object: "response",
    status: "completed",
    model,
    output: [{
      id: "msg_a",
      type: "message",
      role: "assistant",
      status: "completed",
      content: [{ type: "output_text", text, annotations: [] }],
    }],
    usage: { input_tokens: 2, output_tokens: 1, total_tokens: 3 },
  };
}

/** A completed buffered Responses body whose only output item is a tool call. */
function completedWithCall(
  model: string,
  name: string,
  itemType: "function_call" | "custom_tool_call" = "function_call",
): Record<string, unknown> {
  return {
    id: `resp-call-${model}`,
    object: "response",
    status: "completed",
    model,
    output: [{ id: `call_${model}`, type: itemType, name, call_id: `call_${model}`, arguments: "{}" }],
    usage: { input_tokens: 2, output_tokens: 1, total_tokens: 3 },
  };
}

const tools = [{ type: "function", name: "shell", parameters: { type: "object" } }];

function requiredRequest(extra: Record<string, unknown> = {}): Request {
  return new Request("http://localhost/v1/responses", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: "combo/free", input: "hello", stream: false, tools, ...extra }),
  });
}

describe("combo required-tool contract on buffered children (#7456)", () => {
  test("buffered completed with zero call items hops when tool_choice requires a tool", async () => {
    const hits: string[] = [];
    const a = serve(() => {
      hits.push("a");
      return Response.json(completedNoCall("m1", "false success text"));
    });
    const b = serve(() => {
      hits.push("b");
      return Response.json(completedWithCall("m2", "shell"));
    });
    const parent: RequestLogContext = { model: "", provider: "" };
    const response = await handleResponsesWith(comboConfig({
      a: provider("openai-responses", baseUrl(a), "key-a"),
      b: provider("openai-responses", baseUrl(b), "key-b"),
    }), requiredRequest({ tool_choice: "required" }), parent);
    expect(response.status).toBe(200);
    const body = await response.text();
    expect(body).toContain("call_m2");
    expect(body).not.toContain("false success text");
    expect(hits).toEqual(["a", "b"]);
    expect(parent).toMatchObject({
      provider: "combo",
      model: "combo/free",
      resolvedModel: "m2",
      attempts: [
        { ordinal: 1, provider: "a", model: "m1", status: 502 },
        { ordinal: 2, provider: "b", model: "m2" },
      ],
    });
  });

  test("a named tool_choice violation hops; a matching call is accepted", async () => {
    const hits: string[] = [];
    const a = serve(() => {
      hits.push("a");
      return Response.json(completedWithCall("m1", "web"));
    });
    const b = serve(() => {
      hits.push("b");
      return Response.json(completedWithCall("m2", "shell"));
    });
    const response = await handleResponsesWith(comboConfig({
      a: provider("openai-responses", baseUrl(a), "key-a"),
      b: provider("openai-responses", baseUrl(b), "key-b"),
    }), requiredRequest({
      tool_choice: { type: "function", name: "shell" },
      // Declare BOTH tools: with `web` undeclared the existing undeclared-tool guard would be
      // what hops, and the named-choice arm of the contract would go untested.
      tools: [
        { type: "function", name: "shell", parameters: { type: "object" } },
        { type: "function", name: "web", parameters: { type: "object" } },
      ],
    }));
    expect(response.status).toBe(200);
    expect(await response.text()).toContain("call_m2");
    expect(hits).toEqual(["a", "b"]);
  });

  test("a satisfied required contract is accepted on the first target", async () => {
    const hits: string[] = [];
    const a = serve(() => {
      hits.push("a");
      return Response.json(completedWithCall("m1", "shell"));
    });
    const response = await handleResponsesWith(comboConfig({
      a: provider("openai-responses", baseUrl(a), "key-a"),
      b: provider("openai-responses", baseUrl(a), "key-b"),
    }), requiredRequest({ tool_choice: "required" }));
    expect(response.status).toBe(200);
    expect(await response.text()).toContain("call_m1");
    expect(hits).toEqual(["a"]);
  });

  test("a custom_tool_call item satisfies a required contract", async () => {
    const hits: string[] = [];
    const a = serve(() => {
      hits.push("a");
      return Response.json(completedWithCall("m1", "apply_patch", "custom_tool_call"));
    });
    const response = await handleResponsesWith(comboConfig({
      a: provider("openai-responses", baseUrl(a), "key-a"),
      b: provider("openai-responses", baseUrl(a), "key-b"),
    }), requiredRequest({
      tool_choice: "required",
      // The undeclared-tool guard must not be what answers here: declare the custom tool the
      // fixture calls, so only the required-tool contract is under test.
      tools: [{ type: "custom", name: "apply_patch" }],
    }));
    expect(response.status).toBe(200);
    expect(await response.text()).toContain("call_m1");
    expect(hits).toEqual(["a"]);
  });

  test("no tool_choice means the contract is never judged", async () => {
    const hits: string[] = [];
    const a = serve(() => {
      hits.push("a");
      return Response.json(completedNoCall("m1", "ordinary answer"));
    });
    const response = await handleResponsesWith(comboConfig({
      a: provider("openai-responses", baseUrl(a), "key-a"),
      b: provider("openai-responses", baseUrl(a), "key-b"),
    }), requiredRequest());
    expect(response.status).toBe(200);
    expect(await response.text()).toContain("ordinary answer");
    expect(hits).toEqual(["a"]);
  });

  test("every target violating returns the typed compatibility error, not a false completed", async () => {
    const hits: string[] = [];
    const a = serve(() => {
      hits.push("a");
      return Response.json(completedNoCall("m1", "first false success"));
    });
    const b = serve(() => {
      hits.push("b");
      return Response.json(completedNoCall("m2", "second false success"));
    });
    const response = await handleResponsesWith(comboConfig({
      a: provider("openai-responses", baseUrl(a), "key-a"),
      b: provider("openai-responses", baseUrl(b), "key-b"),
    }), requiredRequest({ tool_choice: "required" }));
    expect(hits).toEqual(["a", "b"]);
    expect(response.status).toBe(502);
    const body = await response.text();
    expect(body).toContain("required_tool_contract_unsatisfied");
    expect(body).not.toContain("first false success");
    expect(body).not.toContain("second false success");
  });

  test("an incomplete buffered child stays delivered even with a required contract", async () => {
    const hits: string[] = [];
    const a = serve(() => {
      hits.push("a");
      return Response.json({
        ...completedNoCall("m1", "ran out of budget"),
        status: "incomplete",
        incomplete_details: { reason: "max_output_tokens" },
      });
    });
    const response = await handleResponsesWith(comboConfig({
      a: provider("openai-responses", baseUrl(a), "key-a"),
      b: provider("openai-responses", baseUrl(a), "key-b"),
    }), requiredRequest({ tool_choice: "required" }));
    expect(response.status).toBe(200);
    expect(await response.text()).toContain("ran out of budget");
    expect(hits).toEqual(["a"]);
  });

  test("a streaming completed stream without call items is never re-judged", async () => {
    const hits: string[] = [];
    const sse = (events: unknown[]) => new Response(
      events.map(event => `data: ${JSON.stringify(event)}\n\n`).join(""),
      { headers: { "content-type": "text/event-stream" } },
    );
    const a = serve(() => {
      hits.push("a");
      return sse([
        { type: "response.created", response: { id: "r1", status: "in_progress" } },
        { type: "response.output_text.delta", delta: "streamed text" },
        { type: "response.completed", response: { id: "r1", status: "completed", output: [] } },
      ]);
    });
    const response = await handleResponsesWith(comboConfig({
      a: provider("openai-responses", baseUrl(a), "key-a"),
      b: provider("openai-responses", baseUrl(a), "key-b"),
    }), requiredRequest({ stream: true, tool_choice: "required" }));
    expect(response.status).toBe(200);
    const body = await response.text();
    expect(body).toContain("response.completed");
    expect(body).toContain("streamed text");
    expect(hits).toEqual(["a"]);
  });
});

const { handleResponses } = await import("../../src/server/responses");

async function handleResponsesWith(config: OcxConfig, request: Request, parent?: RequestLogContext) {
  return handleResponses(request, config, parent ?? { model: "", provider: "" });
}
