import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations";
const DEFAULT_MODEL = "doubao-seedream-4.5";
const DEFAULT_SIZE = "1728x2304";
const DEFAULT_TIMEOUT_MS = 120000;
const PLUGIN_ID = "image-this-jimeng45";

export default function register(api: any) {
  registerTools(api);
}

function registerTools(api: any) {
  api.registerTool(
    {
      name: "image_generate",
      description: "Generate an image via Jimeng 4.5 (Ark / Seedream 4.5). Returns an image.",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          prompt: { type: "string", description: "Image prompt" },
          size: { type: "string", description: `Override size (default ${DEFAULT_SIZE})` },
          watermark: { type: "boolean", description: "Override watermark" },
          referenceImages: {
            type: "array",
            items: { type: "string" },
            description:
              "Optional reference images. Accepts http(s) URLs, data:image/*;base64,..., or raw base64 (will be wrapped / uploaded if configured).",
          },
        },
        required: ["prompt"],
      },
      async execute(_id: string, params: Record<string, any> | undefined) {
        try {
          const cfg = getConfig(api);
          const prompt = String(params?.prompt ?? "").trim();
          if (!prompt) {
            throw new Error("prompt is required");
          }

          const size = (params?.size ? String(params.size) : cfg.size) || DEFAULT_SIZE;
          const watermark = typeof params?.watermark === "boolean" ? params.watermark : cfg.watermark;
          const referenceImages: string[] | undefined = Array.isArray(params?.referenceImages)
            ? params!.referenceImages.map((v: any) => String(v || "").trim()).filter(Boolean)
            : undefined;

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
            ? await uploadBase64ToSuperbed(imageBase64, `seedream-${Date.now()}.jpg`, cfg.superbedToken)
            : undefined;

          const textParts = [];
          if (imageUrl) {
            textParts.push(`MEDIA: ${imageUrl}`);
            textParts.push(`![seedream-45](${imageUrl})`);
          }
          textParts.push(JSON.stringify({ ...metadata, imageUrl }, null, 2));

          if (imageUrl) {
            return {
              content: [{ type: "text", text: textParts.join("\n\n") }],
            };
          }

          return {
            content: [
              { type: "image", data: imageBase64, mimeType },
              { type: "text", text: textParts.join("\n\n") },
            ],
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
      "Missing apiKey (plugins.entries.image-this-jimeng45.config.apiKey); or set env JIMENG45_API_KEY/ARK_API_KEY"
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

async function uploadBase64ToSuperbed(base64: string, filename: string, token: string): Promise<string> {
  const cleanBase64 = stripBase64Header(base64);
  const buffer = Buffer.from(cleanBase64, "base64");

  const blob = new Blob([buffer], { type: "image/jpeg" });
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

    if (isHttpUrl(v) || v.startsWith("data:image/")) {
      out.push(v);
      continue;
    }

    if (superbedToken) {
      const url = await uploadBase64ToSuperbed(v, `seedream-${Date.now()}.png`, superbedToken);
      out.push(url);
    } else {
      out.push(`data:image/png;base64,${v}`);
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
    response_format: "b64_json",
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
