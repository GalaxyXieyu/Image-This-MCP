import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations";
const DEFAULT_MODEL = "doubao-seedream-4-5-251128";
const DEFAULT_SIZE = "1728x2304";
const MIN_PIXELS = 3686400;
const DEFAULT_TIMEOUT_MS = 120000;
const DEFAULT_CANDIDATES = 1;
const MAX_CANDIDATES = 8;
const PLUGIN_ID = "img-generator";
const RATIO_PRESETS: Record<string, string> = {
  "1:1": "2048x2048",
  "3:4": "1728x2304",
  "4:3": "2304x1728",
  "16:9": "2560x1440",
  "9:16": "1440x2560",
  "21:9": "2940x1260",
  "9:21": "1260x2940",
};

export default function register(api: any) {
  registerTools(api);
}

function registerTools(api: any) {
  api.registerTool(
    {
      name: "img_generate",
      description:
        "Generate an image (Jimeng 4.5 / Seedream 4.5). Use ratio for aspect ratio (1:1/3:4/4:3/16:9/9:16/21:9) or size (WxH/2K/4K). Output includes MEDIA and Markdown image link when URL is available.",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          prompt: { type: "string", description: "Image prompt" },
          ratio: {
            type: "string",
            enum: ["1:1", "3:4", "4:3", "16:9", "9:16", "21:9", "9:21"],
            description: "Aspect ratio preset. If size is not provided, ratio maps to a valid resolution.",
          },
          size: {
            type: "string",
            description: `Override size. Supports WxH, 2K, 4K. If both size and ratio are provided, size wins (default ${DEFAULT_SIZE}).`,
          },
          watermark: { type: "boolean", description: "Override watermark" },
          referenceImages: {
            type: "array",
            items: { type: "string" },
            description:
              "Optional reference images. Accepts http(s) URLs, data:image/*;base64,..., or raw base64 (will be wrapped / uploaded if configured).",
          },
          candidates: {
            type: "integer",
            description: "Number of candidates to generate (batch).",
            minimum: 1,
            maximum: 8,
          },
        },
        required: ["prompt"],
      },
      async execute(_id: string, params: Record<string, any> | undefined) {
        try {
          const cfg = getConfig(api);

          const candidatesRaw = params?.candidates;
          const candidates = clampInt(
            typeof candidatesRaw === "number" ? candidatesRaw : Number(candidatesRaw),
            DEFAULT_CANDIDATES,
            1,
            MAX_CANDIDATES
          );

          const ratio = String(params?.ratio || "").trim();
          const ratioSize = ratio ? sizeFromRatioPreset(ratio) : undefined;
          const requestedSize = (params?.size ? String(params.size) : ratioSize || cfg.size) || DEFAULT_SIZE;
          const sizeInfo = normalizeSize(requestedSize, DEFAULT_SIZE);
          const size = sizeInfo.size;
          const watermark = typeof params?.watermark === "boolean" ? params.watermark : cfg.watermark;
          const referenceImages: string[] | undefined = normalizeReferenceImagesInput(params?.referenceImages);

          const prompt = String(params?.prompt ?? "").trim();
          if (!prompt) {
            throw new Error("prompt is required");
          }

          const jobId = crypto.randomUUID();
          const startedAt = new Date().toISOString();

          const images: any[] = [];
          for (let i = 0; i < candidates; i++) {
            // Note: Seed is not wired into Ark image generations here yet; reserved for future.
            const { imageBase64, mimeType, metadata } = await generateSeedream45({
              apiKey: cfg.apiKey,
              baseUrl: cfg.baseUrl,
              model: cfg.model,
              size,
              watermark,
              timeoutMs: cfg.timeoutMs,
              superbedToken: cfg.superbedToken,
              prompt,
              referenceImages,
            });

            const imageUrl = cfg.superbedToken
              ? await uploadBase64ToSuperbed(imageBase64, `seedream-${Date.now()}-${i + 1}.jpg`, cfg.superbedToken)
              : undefined;
            const providerUrl = typeof metadata?.url === "string" ? metadata.url : undefined;
            const renderUrl = imageUrl || providerUrl;

            images.push({
              index: i + 1,
              renderUrl,
              imageUrl,
              providerUrl,
              mimeType,
              metadata,
            });
          }

          const finishedAt = new Date().toISOString();
          const resultObj = {
            jobId,
            tool: "img_generate",
            provider: "img-generator",
            model: cfg.model,
            size,
            watermark,
            ratioRequested: ratio || undefined,
            sizeRequested: sizeInfo.requested,
            sizeAdjusted: sizeInfo.adjusted,
            sizeNote: sizeInfo.note,
            candidates,
            prompt,
            startedAt,
            finishedAt,
            outputs: images.map((img) => ({
              index: img.index,
              url: img.renderUrl,
              imageUrl: img.imageUrl,
              providerUrl: img.providerUrl,
              mimeType: img.mimeType,
            })),
          };

          // Best-effort job logging (jsonl). Never log secrets.
          try {
            appendJobLog(resultObj);
          } catch {
            // ignore logging failures
          }

          const textParts: string[] = [];
          // Human-friendly media list first
          for (const out of resultObj.outputs) {
            if (out.url) {
              textParts.push(`MEDIA: ${out.url}`);
              textParts.push(`![seedream](${out.url})`);
            }
          }
          textParts.push(JSON.stringify(resultObj, null, 2));

          return {
            content: [{ type: "text", text: textParts.join("\n\n") }],
          };
        } catch (error: any) {
          const message = error instanceof Error ? error.message : String(error);
          return {
            isError: true,
            content: [{ type: "text", text: `image_generate failed: ${message}` }],
          };
        }
      },
    },
    { optional: true }
  );
}

function clampInt(value: number, fallback: number, min: number, max: number) {
  const n = Number.isFinite(value) ? Math.trunc(value) : fallback;
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}

// Removed: structured prompt helpers (negative/typography/promptSpec) to align tool inputs with upstream API (prompt-only).

function getWorkspaceDir() {
  const fromEnv = String(process.env.OPENCLAW_WORKSPACE || "").trim();
  if (fromEnv) return fromEnv;

  // Try to infer from OpenClaw config if available.
  try {
    const configPath =
      process.env.OPENCLAW_CONFIG_PATH || path.join(os.homedir(), ".openclaw", "openclaw.json");
    if (fs.existsSync(configPath)) {
      const raw = fs.readFileSync(configPath, "utf8");
      const parsed = JSON.parse(raw);
      const ws =
        String(parsed?.agents?.defaults?.workspace || "").trim() ||
        String(parsed?.agents?.list?.find((a: any) => a?.id === "main")?.workspace || "").trim();
      if (ws) return ws;
    }
  } catch {
    // ignore
  }

  return path.join(os.homedir(), ".openclaw", "workspace");
}

function appendJobLog(job: any) {
  const workspace = getWorkspaceDir();
  const dir = path.join(workspace, "image-jobs");
  fs.mkdirSync(dir, { recursive: true });
  const logPath = path.join(dir, "img-generator.jsonl");

  // Avoid leaking huge strings
  const safeJob = { ...job };
  if (safeJob.prompt && String(safeJob.prompt).length > 5000) {
    safeJob.prompt = String(safeJob.prompt).slice(0, 5000) + "…";
  }

  fs.appendFileSync(logPath, JSON.stringify(safeJob) + "\n", "utf8");
}

function loadConfigFallback() {
  try {
    const configPath =
      process.env.OPENCLAW_CONFIG_PATH ||
      path.join(os.homedir(), ".openclaw", "openclaw.json");
    const raw = fs.readFileSync(configPath, "utf8");
    const parsed = JSON.parse(raw);
    return (
      (parsed &&
        parsed.plugins &&
        parsed.plugins.entries &&
        parsed.plugins.entries[PLUGIN_ID] &&
        parsed.plugins.entries[PLUGIN_ID].config) ||
      {}
    );
  } catch {
    return {};
  }
}

function resolveConfigFromApi(api: any) {
  const root = (api && api.config) || {};
  if (root && typeof root === "object" && root.plugins?.entries) {
    const pluginId = api?.id || api?.pluginId || api?.manifest?.id || PLUGIN_ID;
    const entry = root.plugins.entries?.[pluginId];
    const pluginCfg = entry && entry.config;
    if (pluginCfg && typeof pluginCfg === "object") {
      return pluginCfg;
    }
  }
  if (root && typeof root === "object" && !Object.prototype.hasOwnProperty.call(root, "plugins")) {
    return root;
  }
  return {};
}

function mergeDefined(base: Record<string, any>, override: Record<string, any>) {
  const merged: Record<string, any> = { ...(base || {}) };
  if (override && typeof override === "object") {
    for (const [key, value] of Object.entries(override)) {
      if (value !== undefined && value !== null && value !== "") {
        merged[key] = value;
      }
    }
  }
  return merged;
}

function getConfig(api: any) {
  const cfgFromApi = resolveConfigFromApi(api);
  const cfg = mergeDefined(loadConfigFallback(), cfgFromApi);

  const apiKey =
    String(cfg.apiKey || "").trim() ||
    String(process.env.JIMENG45_API_KEY || "").trim() ||
    String(process.env.ARK_API_KEY || "").trim() ||
    String(process.env.VOLCES_ARK_API_KEY || "").trim();

  if (!apiKey) {
    throw new Error(
      "Missing apiKey (plugins.entries.img-generator.config.apiKey); or set env JIMENG45_API_KEY/ARK_API_KEY"
    );
  }

  const envBaseUrl = String(process.env.JIMENG45_BASE_URL || "").trim();
  const envModel = String(process.env.JIMENG45_MODEL || "").trim();
  const envSize = String(process.env.JIMENG45_SIZE || "").trim();
  const envTimeoutMsRaw = String(process.env.JIMENG45_TIMEOUT_MS || "").trim();
  const envWatermarkRaw = String(process.env.JIMENG45_WATERMARK || "").trim().toLowerCase();
  const envSuperbedToken =
    String(process.env.JIMENG45_SUPERBED_TOKEN || "").trim() ||
    String(process.env.SUPERBED_TOKEN || "").trim();

  const envTimeoutMs = envTimeoutMsRaw ? Number(envTimeoutMsRaw) : undefined;
  const envWatermark = envWatermarkRaw === "true" ? true : envWatermarkRaw === "false" ? false : undefined;

  return {
    apiKey,
    baseUrl: (String(cfg.baseUrl || "").trim() || envBaseUrl || DEFAULT_BASE_URL).trim() || DEFAULT_BASE_URL,
    model: (String(cfg.model || "").trim() || envModel || DEFAULT_MODEL).trim() || DEFAULT_MODEL,
    size: (String(cfg.size || "").trim() || envSize || DEFAULT_SIZE).trim() || DEFAULT_SIZE,
    watermark: typeof cfg.watermark === "boolean" ? cfg.watermark : typeof envWatermark === "boolean" ? envWatermark : false,
    timeoutMs: normalizeTimeout(typeof cfg.timeoutMs !== "undefined" ? cfg.timeoutMs : envTimeoutMs),
    superbedToken: cfg.superbedToken ? String(cfg.superbedToken).trim() : envSuperbedToken,
  };
}

function normalizeTimeout(value: any) {
  if (typeof value === "number" && Number.isFinite(value) && value >= 1000) {
    return Math.min(value, 180000);
  }
  return DEFAULT_TIMEOUT_MS;
}

function isHttpUrl(value: string) {
  return /^https?:\/\//i.test(value);
}

function stripBase64Header(base64: string) {
  return base64.replace(/^data:image\/[^;]+;base64,/, "");
}

function normalizeBase64(value: string) {
  return String(value || "")
    .trim()
    .replace(/\s+/g, "")
    .replace(/-/g, "+")
    .replace(/_/g, "/");
}

function isValidBase64(value: string) {
  const clean = normalizeBase64(value);
  if (!clean) return false;
  if (!/^[A-Za-z0-9+/=]+$/.test(clean)) return false;
  if (clean.length % 4 === 1) return false;
  if (clean.includes("=") && !/={0,2}$/.test(clean)) return false;
  return true;
}

function parseDataUrl(value: string) {
  const match = String(value || "").trim().match(/^data:(image\/[^;]+);base64,(.*)$/i);
  if (!match) return null;
  const mimeType = match[1];
  const base64 = normalizeBase64(match[2]);
  if (!isValidBase64(base64)) {
    throw new Error("Invalid base64 image data URL");
  }
  return {
    mimeType,
    base64,
    dataUrl: `data:${mimeType};base64,${base64}`,
  };
}

function guessMimeType(filePath: string) {
  const ext = path.extname(filePath || "").toLowerCase();
  if (ext === ".jpg" || ext === ".jpeg") return "image/jpeg";
  if (ext === ".png") return "image/png";
  if (ext === ".webp") return "image/webp";
  if (ext === ".gif") return "image/gif";
  if (ext === ".bmp") return "image/bmp";
  if (ext === ".heic") return "image/heic";
  return "image/png";
}

function expandHomeDir(filePath: string) {
  if (filePath.startsWith("~/")) {
    return path.join(os.homedir(), filePath.slice(2));
  }
  return filePath;
}

function resolveLocalFile(value: string) {
  let filePath = String(value || "").trim();
  if (!filePath) return null;
  try {
    if (filePath.startsWith("file://")) {
      filePath = fileURLToPath(filePath);
    }
  } catch {
    return null;
  }
  filePath = expandHomeDir(filePath);
  try {
    if (!fs.existsSync(filePath)) return null;
    const stat = fs.statSync(filePath);
    if (!stat.isFile()) return null;
  } catch {
    return null;
  }
  const bytes = fs.readFileSync(filePath);
  const mimeType = guessMimeType(filePath);
  const base64 = bytes.toString("base64");
  return {
    filePath,
    mimeType,
    base64,
    dataUrl: `data:${mimeType};base64,${base64}`,
    filename: path.basename(filePath),
  };
}

function normalizeSize(requested: string, fallback: string) {
  const raw = String(requested || "").trim();
  if (!raw) {
    return { size: fallback, requested: raw, adjusted: true, note: "size empty; fallback applied" };
  }
  const normalizedAlias = raw.toLowerCase();
  if (normalizedAlias === "2k") {
    return { size: "2048x2048", requested: raw, adjusted: true, note: "size alias 2K mapped to 2048x2048" };
  }
  if (normalizedAlias === "4k") {
    return { size: "4096x4096", requested: raw, adjusted: true, note: "size alias 4K mapped to 4096x4096" };
  }
  const match = raw.match(/^(\d+)\s*x\s*(\d+)$/i);
  if (!match) {
    return { size: fallback, requested: raw, adjusted: true, note: "size format invalid; fallback applied" };
  }
  const width = Number(match[1]);
  const height = Number(match[2]);
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return { size: fallback, requested: raw, adjusted: true, note: "size invalid; fallback applied" };
  }
  const pixels = width * height;
  if (pixels < MIN_PIXELS) {
    return {
      size: fallback,
      requested: raw,
      adjusted: true,
      note: `size too small (${width}x${height}); min pixels ${MIN_PIXELS}; fallback applied`,
    };
  }
  return { size: `${width}x${height}`, requested: raw, adjusted: false, note: undefined };
}

function sizeFromRatioPreset(ratio: string) {
  const normalized = String(ratio || "").trim();
  return RATIO_PRESETS[normalized];
}

function normalizeReferenceImagesInput(value: any) {
  if (Array.isArray(value)) {
    return value.map((v) => String(v || "").trim()).filter(Boolean);
  }
  if (typeof value === "string") {
    const v = value.trim();
    return v ? [v] : undefined;
  }
  return undefined;
}

async function uploadBase64ToSuperbed(
  base64: string,
  filename: string,
  token: string,
  mimeType?: string
): Promise<string> {
  const cleanBase64 = normalizeBase64(stripBase64Header(base64));
  const buffer = Buffer.from(cleanBase64, "base64");

  const parsedDataUrl = parseDataUrl(base64);
  const resolvedMimeType = mimeType || parsedDataUrl?.mimeType || "image/jpeg";
  const blob = new Blob([buffer], { type: resolvedMimeType });
  const formData = new FormData();
  formData.append("file", blob, filename.replace(/\.png$/i, ".jpg"));

  const res = await fetch(`https://api.superbed.cn/upload?token=${encodeURIComponent(token)}`, {
    method: "POST",
    body: formData,
  });

  const result: any = await res.json();
  if (result?.err !== 0 || !result?.url) {
    throw new Error(`superbed upload failed: ${result?.msg || "unknown"}`);
  }

  const superbedUrl = String(result.url);
  try {
    const head = await fetch(superbedUrl, { method: "HEAD", redirect: "manual" as any });
    const direct = head.headers.get("location");
    if (direct && direct.startsWith("http")) return direct;
  } catch {
    // ignore
  }
  return superbedUrl;
}

async function normalizeReferenceImages(images: string[] | undefined, superbedToken: string) {
  if (!Array.isArray(images) || images.length === 0) return undefined;

  const out: string[] = [];
  for (const raw of images) {
    const v = String(raw || "").trim();
    if (!v) continue;

    if (isHttpUrl(v)) {
      out.push(v);
      continue;
    }

    const dataUrlInfo = parseDataUrl(v);
    if (dataUrlInfo) {
      if (superbedToken) {
        const url = await uploadBase64ToSuperbed(
          dataUrlInfo.base64,
          `seedream-${Date.now()}.png`,
          superbedToken,
          dataUrlInfo.mimeType
        );
        out.push(url);
      } else {
        out.push(dataUrlInfo.dataUrl);
      }
      continue;
    }

    const localFile = resolveLocalFile(v);
    if (localFile) {
      if (superbedToken) {
        const url = await uploadBase64ToSuperbed(
          localFile.base64,
          localFile.filename || `seedream-${Date.now()}.png`,
          superbedToken,
          localFile.mimeType
        );
        out.push(url);
      } else {
        out.push(localFile.dataUrl);
      }
      continue;
    }

    const cleaned = normalizeBase64(v);
    if (!isValidBase64(cleaned)) {
      throw new Error("Invalid reference image: expected URL, data URL, base64, or readable local file path");
    }

    if (superbedToken) {
      const url = await uploadBase64ToSuperbed(cleaned, `seedream-${Date.now()}.png`, superbedToken);
      out.push(url);
    } else {
      out.push(`data:image/png;base64,${cleaned}`);
    }
  }

  if (out.length === 0) return undefined;
  return out.length === 1 ? out[0] : out;
}

async function fetchImageAsBase64(url: string, timeoutMs: number): Promise<{ base64: string; mimeType: string }> {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { method: "GET", signal: controller.signal });
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`image download failed: HTTP ${res.status} ${body}`);
    }
    const mimeType = res.headers.get("content-type") || "image/jpeg";
    const ab = await res.arrayBuffer();
    const base64 = Buffer.from(ab).toString("base64");
    return { base64, mimeType };
  } finally {
    clearTimeout(t);
  }
}

async function postJson(url: string, body: any, headers: Record<string, string>, timeoutMs: number) {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...headers,
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`HTTP ${res.status}: ${text}`);
    }

    return res.json();
  } finally {
    clearTimeout(t);
  }
}

async function generateSeedream45(opts: {
  apiKey: string;
  baseUrl: string;
  model: string;
  size: string;
  watermark: boolean;
  timeoutMs: number;
  superbedToken: string;
  prompt: string;
  referenceImages?: string[];
}): Promise<{ imageBase64: string; mimeType: string; metadata: any }> {
  const imagePayload = await normalizeReferenceImages(opts.referenceImages, opts.superbedToken);

  const requestBody: Record<string, any> = {
    model: opts.model,
    prompt: opts.prompt,
    response_format: "url",
    sequential_image_generation: "disabled",
    size: opts.size,
    stream: false,
    watermark: !!opts.watermark,
  };

  if (imagePayload) {
    requestBody.image = imagePayload;
  }

  const result: any = await postJson(
    opts.baseUrl,
    requestBody,
    {
      Authorization: `Bearer ${opts.apiKey}`,
    },
    opts.timeoutMs
  );

  if (result?.error) {
    const message = result.error?.message || "Seedream API error";
    throw new Error(`Seedream 4.5 call failed: ${message}`);
  }

  const dataList = Array.isArray(result?.data) ? result.data : [];
  if (dataList.length === 0) {
    throw new Error("Seedream 4.5 returned empty data");
  }

  const item = dataList.find((e: any) => e?.b64_json || e?.url || e?.error) || dataList[0];
  if (item?.error) {
    const code = item.error?.code || "unknown";
    const message = item.error?.message || "Unknown";
    throw new Error(`Seedream 4.5 generation failed: code=${code}, msg=${message}`);
  }

  if (item?.b64_json) {
    return {
      imageBase64: String(item.b64_json),
      mimeType: "image/jpeg",
      metadata: {
        provider: "jimeng-45",
        model: result?.model || opts.model,
        size: item?.size || opts.size,
        created: result?.created,
        usage: result?.usage,
      },
    };
  }

  if (item?.url) {
    const downloaded = await fetchImageAsBase64(String(item.url), opts.timeoutMs);
    return {
      imageBase64: downloaded.base64,
      mimeType: downloaded.mimeType,
      metadata: {
        provider: "jimeng-45",
        model: result?.model || opts.model,
        size: item?.size || opts.size,
        created: result?.created,
        usage: result?.usage,
        url: item.url,
      },
    };
  }

  throw new Error("Seedream 4.5 response missing image data");
}
