import { REQUIRED_TOOL_CONTRACT_CODE } from "../../lib/errors";
import { carryResponseToolOwnership } from "./response-tool-ownership";

/**
 * The required-tool contract on a buffered combo child (#7456).
 *
 * A child that answers `tool_choice`-required with `status: "completed"` and zero call items is
 * a false success: the Responses consumer sees a terminal it believes needs no continuation, and
 * the agent loop ends with nothing to execute. The combo accepted any 200 before commit, so the
 * hop machinery — reachable only from a non-OK response — never saw it. This module judges the
 * one shape where a hop is provably safe: the fully buffered non-streaming body, before the
 * child is committed. A streamed text-only completion stays delivered on purpose; replaying a
 * stream after visible output is not safe, so the stream preflight's conservative zero-output
 * rules remain the only judgment there.
 */

/**
 * `tool_choice` demanding a call: any declared tool — carrying the hosted-call item types the
 * request's declared hosted tools permit — or one named function/custom tool.
 */
export type RequiredToolContract =
  | { kind: "any"; hostedCallTypes: ReadonlySet<string> }
  | { kind: "named"; name: string };

/**
 * What the request requires of the child, or undefined when the request requires no tool call
 * and there is nothing to judge. Only an explicit `required` or a named function/custom choice
 * demands one; `auto`, `none`, `null` and malformed shapes never do, and a request with no
 * declared tools cannot be satisfied by any response, so it is not judged either.
 */
export function requiredToolContractOf(rawBody: unknown): RequiredToolContract | undefined {
  if (!rawBody || typeof rawBody !== "object" || Array.isArray(rawBody)) return undefined;
  const body = rawBody as Record<string, unknown>;
  const tools = body.tools;
  if (!Array.isArray(tools) || tools.length === 0) return undefined;
  const choice = body.tool_choice;
  if (choice === "required") {
    const hostedCallTypes = hostedCallTypesOf(tools);
    // A capability this module cannot map to an output item type makes the demand
    // unprovable, so the request is not judged at all.
    return hostedCallTypes === undefined ? undefined : { kind: "any", hostedCallTypes };
  }
  if (choice && typeof choice === "object" && !Array.isArray(choice)) {
    const named = choice as Record<string, unknown>;
    if ((named.type === "function" || named.type === "custom")
      && typeof named.name === "string" && named.name.trim().length > 0) {
      return { kind: "named", name: named.name };
    }
  }
  return undefined;
}

/** Response output item types that execute a declared tool on the client. */
const CALL_ITEM_TYPES = new Set(["function_call", "custom_tool_call"]);

/**
 * Hosted-tool DECLARATION type -> the built-in output item type executing it produces.
 * Canonical Responses forwarding preserves the client's hosted-tool declarations into every
 * combo child, so a child may honor `required` by executing one itself — but only a declared
 * capability counts, or a function-only catalog could return an arbitrary hosted call and skip
 * its real work on every target. Alias spellings fold onto one capability, the same grouping
 * `hosted-tool-policy` uses for `unsupportedHostedTools`.
 */
const HOSTED_TOOL_CALL_ITEM_TYPES = new Map<string, string>([
  ["web_search", "web_search_call"],
  ["web_search_preview", "web_search_call"],
  ["file_search", "file_search_call"],
  ["code_interpreter", "code_interpreter_call"],
  ["image_generation", "image_generation_call"],
  ["image_gen", "image_generation_call"],
  ["mcp", "mcp_call"],
  ["computer_use", "computer_call"],
  ["computer_use_preview", "computer_call"],
]);

/**
 * Declarable hosted types (hosted-tool-policy's closed vocabulary) whose output item spelling
 * this module cannot prove. A request declaring one of these is not judged: the module's
 * invariant is that only provable violations ever reach the failure path.
 */
const UNMAPPABLE_HOSTED_DECLARATIONS = new Set(["x_search", "tool_search", "local_shell"]);

/** The hosted-call item types the declared tools permit, or undefined when judgment must abstain. */
function hostedCallTypesOf(tools: readonly unknown[]): ReadonlySet<string> | undefined {
  const out = new Set<string>();
  for (const tool of tools) {
    if (!tool || typeof tool !== "object" || Array.isArray(tool)) continue;
    const type = (tool as Record<string, unknown>).type;
    if (typeof type !== "string") continue;
    const callType = HOSTED_TOOL_CALL_ITEM_TYPES.get(type);
    if (callType) {
      out.add(callType);
    } else if (UNMAPPABLE_HOSTED_DECLARATIONS.has(type)) {
      return undefined;
    }
  }
  return out;
}

/**
 * Whether the completed payload carries the demanded call. A missing or non-array `output`,
 * and a named choice carried by an item with no name to read, stay SATISFIED: this judges only
 * what it can prove violated, and everything unprovable keeps its current delivery.
 */
function carriesRequiredCall(payload: Record<string, unknown>, requirement: RequiredToolContract): boolean {
  const output = payload.output;
  if (!Array.isArray(output)) return true;
  for (const item of output) {
    if (!item || typeof item !== "object" || Array.isArray(item)) continue;
    const call = item as Record<string, unknown>;
    if (typeof call.type !== "string") continue;
    // A hosted item satisfies the demand for SOME call only when its capability was declared;
    // it can never satisfy a demand that named a specific function/custom tool.
    if (requirement.kind === "any" && requirement.hostedCallTypes.has(call.type)) return true;
    if (!CALL_ITEM_TYPES.has(call.type)) continue;
    if (requirement.kind === "any") return true;
    if (typeof call.name !== "string" || call.name === requirement.name) return true;
  }
  return false;
}

function violationResponse(requirement: RequiredToolContract): Response {
  const demanded = requirement.kind === "any"
    ? "a tool call"
    : `a call to the tool "${requirement.name}"`;
  return new Response(JSON.stringify({
    error: {
      type: "upstream_error",
      code: REQUIRED_TOOL_CONTRACT_CODE,
      message: `Target completed the response without ${demanded} although tool_choice required it`,
    },
  }), { status: 502, headers: { "Content-Type": "application/json" } });
}

export type BufferedRequiredToolVerdict =
  | { kind: "accepted"; response: Response }
  | { kind: "violation"; response: Response };

/**
 * Judge a buffered (application/json) 200 Responses child against the request's required-tool
 * contract. Returns either a violation failure for the combo's ordinary failure path — the body
 * is fully buffered and uncommitted, so an ordinary hop costs the client nothing — or the
 * response rebuilt from the consumed body with its identity markers carried over, since reading
 * the body is destructive. Only `status: "completed"` payloads can violate: `incomplete` and
 * `failed` carry their own reasons and reach the client today. A body that cannot be read or
 * parsed violates too: it could not be delivered either, and the failure path at least classifies
 * it with a status instead of forwarding bytes as success.
 */
export async function judgeBufferedRequiredTool(
  response: Response,
  requirement: RequiredToolContract,
): Promise<BufferedRequiredToolVerdict> {
  let text: string;
  try {
    text = await response.text();
  } catch {
    return { kind: "violation", response: violationResponse(requirement) };
  }
  let payload: Record<string, unknown> | undefined;
  try {
    const parsed = JSON.parse(text) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) payload = parsed as Record<string, unknown>;
  } catch {
    payload = undefined;
  }
  if (payload?.status === "completed" && !carriesRequiredCall(payload, requirement)) {
    return { kind: "violation", response: violationResponse(requirement) };
  }
  // `text()` decoded the body, so representation headers copied from a compressed original
  // would describe bytes this plain-text Response no longer carries.
  const headers = new Headers(response.headers);
  headers.delete("content-encoding");
  headers.delete("content-length");
  const rebuilt = new Response(text, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
  return { kind: "accepted", response: carryResponseToolOwnership(response, rebuilt) };
}
