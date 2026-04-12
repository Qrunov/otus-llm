import * as vscode from "vscode";
import * as path from "path";
import * as cp from "child_process";

function toPosixRelative(workspaceRoot: string, fsPath: string): string {
  let rel = path.relative(workspaceRoot, fsPath);
  rel = rel.split(path.sep).join("/");
  if (!rel || rel.startsWith("..")) {
    throw new Error(`File is outside workspace: ${fsPath}`);
  }
  return rel;
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

type ExplainResult = {
  ok?: boolean;
  explanation?: string;
  errors?: string[];
  artifact?: string;
};

let outChannel: vscode.OutputChannel;

function logAstrag(message: string, showPanel = false) {
  const line = `[${new Date().toISOString()}] ${message}`;
  outChannel.appendLine(line);
  if (showPanel) {
    outChannel.show(true);
  }
}

function logAstragError(e: unknown) {
  if (e instanceof Error) {
    logAstrag(e.stack ?? e.message, true);
  } else {
    logAstrag(String(e), true);
  }
}

function showExplainPanel(parsed: ExplainResult) {
  const panel = vscode.window.createWebviewPanel(
    "astragExplanation",
    "Astrag explanation",
    vscode.ViewColumn.Beside,
    { enableScripts: false }
  );
  const body = escapeHtml(parsed.explanation ?? "(empty)");
  const errs = (parsed.errors ?? []).map(escapeHtml).join("<br/>");
  const art = escapeHtml(parsed.artifact ?? "");
  panel.webview.html = `<!DOCTYPE html>
<html><head><meta charset="UTF-8"><style>
  body { font-family: var(--vscode-font-family); color: var(--vscode-foreground); padding: 12px; }
  pre.explain { white-space: pre-wrap; word-break: break-word; }
  .meta { font-size: 0.85em; opacity: 0.85; margin-top: 1em; }
  .err { color: var(--vscode-errorForeground); }
</style></head><body>
  <h2>Explanation</h2>
  <pre class="explain">${body}</pre>
  ${errs ? `<p class="err"><strong>Errors</strong><br/>${errs}</p>` : ""}
  <p class="meta">Artifact: ${art}</p>
</body></html>`;
}

function resolvePythonInterpreter(cfg: vscode.WorkspaceConfiguration): string {
  const fromAstrag = (cfg.get<string>("pythonPath") ?? "").trim();
  if (fromAstrag) {
    return fromAstrag;
  }
  const fromPyExt = vscode.workspace
    .getConfiguration("python")
    .get<string>("defaultInterpreterPath");
  if (fromPyExt && fromPyExt.trim()) {
    return fromPyExt.trim();
  }
  return process.platform === "win32" ? "python" : "python3";
}

function runCliExplain(
  configAbs: string,
  cliCwd: string,
  payload: object,
  launch: { kind: "python"; python: string } | { kind: "exe"; executable: string },
  timeoutMs: number
): Promise<ExplainResult> {
  const args =
    launch.kind === "python"
      ? ["-m", "astrag.bench.cli", "vscode-explain", "--config", configAbs]
      : ["vscode-explain", "--config", configAbs];
  const executable = launch.kind === "python" ? launch.python : launch.executable;

  return new Promise((resolve, reject) => {
    let settled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const finish = (fn: () => void) => {
      if (settled) {
        return;
      }
      settled = true;
      if (timer !== undefined) {
        clearTimeout(timer);
      }
      fn();
    };

    const child = cp.spawn(executable, args, {
      cwd: cliCwd,
      env: process.env,
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout?.on("data", (c) => (stdout += c.toString()));
    child.stderr?.on("data", (c) => (stderr += c.toString()));

    timer = setTimeout(() => {
      finish(() => {
        try {
          child.kill("SIGTERM");
        } catch {
          /* ignore */
        }
        setTimeout(() => {
          try {
            child.kill("SIGKILL");
          } catch {
            /* ignore */
          }
        }, 4000);
        reject(
          new Error(
            `Astrag CLI timeout after ${timeoutMs} ms — проверь LLM (OPENAI_BASE_URL) и лог сервера / Output → Astrag`
          )
        );
      });
    }, timeoutMs);

    child.on("error", (e) =>
      finish(() => {
        reject(e);
      })
    );
    child.stdin?.write(JSON.stringify(payload), "utf8");
    child.stdin?.end();
    child.on("close", (code) => {
      if (code !== null && code !== 0) {
        finish(() => {
          reject(
            new Error(`Astrag exited ${code}. ${stderr.slice(-500) || stdout.slice(-500)}`)
          );
        });
        return;
      }
      finish(() => {
        try {
          const line = stdout.trim().split(/\r?\n/).pop() ?? "";
          resolve(JSON.parse(line) as ExplainResult);
        } catch (e) {
          reject(
            new Error(`bad JSON — ${String(e)}. Output: ${stdout.slice(0, 400)}`)
          );
        }
      });
    });
  });
}

async function runHttpExplain(
  baseUrl: string,
  token: string,
  payload: object,
  timeoutMs: number
): Promise<ExplainResult> {
  const url = `${baseUrl.replace(/\/$/, "")}/v1/explain`;
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) {
    headers["X-Astrag-Token"] = token;
  }
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  let r: Response;
  try {
    r = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
      signal: ctrl.signal,
    });
  } catch (e) {
    if (e instanceof Error && e.name === "AbortError") {
      throw new Error(
        `HTTP timeout after ${timeoutMs} ms — сервер или LLM не ответили вовремя (см. Output → Astrag)`
      );
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
  const text = await r.text();
  if (!r.ok) {
    let detail = text.slice(0, 800);
    try {
      const j = JSON.parse(text) as { detail?: string | unknown };
      if (j.detail !== undefined) {
        detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
      }
    } catch {
      /* use raw text */
    }
    throw new Error(`HTTP ${r.status}: ${detail}`);
  }
  try {
    return JSON.parse(text) as ExplainResult;
  } catch (e) {
    throw new Error(`Ответ не JSON — ${String(e)}: ${text.slice(0, 400)}`);
  }
}

export function activate(context: vscode.ExtensionContext) {
  outChannel = vscode.window.createOutputChannel("Astrag");
  context.subscriptions.push(outChannel);

  const disposable = vscode.commands.registerCommand("astrag.explainSelection", async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showWarningMessage("Astrag: open a file first.");
      return;
    }
    const sel = editor.selection;
    if (sel.isEmpty) {
      vscode.window.showWarningMessage("Astrag: select code to explain.");
      return;
    }
    const doc = editor.document;
    const folder = vscode.workspace.getWorkspaceFolder(doc.uri);
    if (!folder) {
      vscode.window.showErrorMessage("Astrag: file must be inside a workspace folder.");
      return;
    }
    const workspaceRoot = folder.uri.fsPath;
    const cfg = vscode.workspace.getConfiguration("astrag");
    const mode = (cfg.get<string>("mode") ?? "http").toLowerCase();
    const configPath = cfg.get<string>("configPath") ?? "configs/default.yaml";
    const cliCommand = cfg.get<string>("cliCommand") ?? "astrag";
    const launchViaPython = cfg.get<boolean>("launchViaPython") ?? true;
    const cliCwd = (cfg.get<string>("cliCwd") ?? "").trim() || workspaceRoot;
    const sourceRoot = (cfg.get<string>("sourceRoot") ?? "").trim();
    const projectBase = sourceRoot ? path.join(workspaceRoot, sourceRoot) : workspaceRoot;
    const serverUrl = (cfg.get<string>("serverUrl") ?? "http://127.0.0.1:8765").trim();
    const serverToken = (cfg.get<string>("serverToken") ?? "").trim();
    const requestTimeoutMs = Math.max(
      5000,
      Number(cfg.get<number>("requestTimeoutMs") ?? 600_000)
    );

    const configAbs = path.isAbsolute(configPath)
      ? configPath
      : path.join(cliCwd, configPath);

    const start = sel.start;
    const end = sel.end;
    const orderedStart = start.isBefore(end) ? start : end;
    const orderedEnd = start.isBefore(end) ? end : start;
    const text = doc.getText(new vscode.Selection(orderedStart, orderedEnd));

    const relFile = toPosixRelative(projectBase, doc.uri.fsPath);
    const line1Based = orderedStart.line + 1;
    const endLine1Based = orderedEnd.line + 1;

    const payload = {
      file: relFile,
      line: line1Based,
      selected_text: text,
      selection_start_line: line1Based,
      selection_start_character: orderedStart.character,
      selection_end_line: endLine1Based,
      selection_end_character: orderedEnd.character,
      experiment_id: "vscode",
    };

    await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title:
          mode === "http"
            ? "Astrag: requesting explain from server…"
            : "Astrag: running explain…",
        cancellable: false,
      },
           async () => {
        try {
          logAstrag(`mode=${mode} file=${relFile}:${line1Based}`);
          let parsed: ExplainResult;
          if (mode === "http") {
            logAstrag(`POST ${serverUrl.replace(/\/$/, "")}/v1/explain`);
            parsed = await runHttpExplain(serverUrl, serverToken, payload, requestTimeoutMs);
          } else {
            const launch = launchViaPython
              ? { kind: "python" as const, python: resolvePythonInterpreter(cfg) }
              : { kind: "exe" as const, executable: cliCommand };
            logAstrag(
              `cli ${launch.kind === "python" ? launch.python + " -m astrag.bench.cli" : launch.executable} vscode-explain --config ${configAbs}`
            );
            parsed = await runCliExplain(configAbs, cliCwd, payload, launch, requestTimeoutMs);
          }
          if (parsed.errors?.length) {
            logAstrag(`server/explain errors: ${parsed.errors.join("; ")}`, true);
          }
          showExplainPanel(parsed);
        } catch (e) {
          logAstragError(e);
          vscode.window.showErrorMessage(
            `Astrag: ${String(e)} — см. панель Output → Astrag`
          );
          throw e;
        }
      }
    );
  });

  context.subscriptions.push(disposable);

  context.subscriptions.push(
    vscode.commands.registerCommand("astrag.showOutput", () => {
      outChannel.show(true);
    })
  );
}

export function deactivate() {}
