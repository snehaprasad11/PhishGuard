import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

const here = import.meta.dirname;

// Files copied into the build unchanged. The model, its metadata and the
// known-sites list ship inside the extension, so analysing a URL never needs
// a network request.
const STATIC_ASSETS: Record<string, string> = {
  "manifest.json": "manifest.json",
  "models/url-model.onnx": "../models/url-model.onnx",
  "models/url-model.json": "../models/url-model.json",
  "models/known-sites.txt": "../models/known-sites.txt",
  "models/text-model.onnx": "../models/text-model.onnx",
  "models/text-model.json": "../models/text-model.json",
  "models/model_card.html": "../models/model_card.html",
  "icons/icon16.png": "icons/icon16.png",
  "icons/icon48.png": "icons/icon48.png",
  "icons/icon128.png": "icons/icon128.png",
};

function copyStaticAssets(): Plugin {
  return {
    name: "copy-static-assets",
    generateBundle() {
      for (const [fileName, source] of Object.entries(STATIC_ASSETS)) {
        this.emitFile({ type: "asset", fileName, source: readFileSync(resolve(here, source)) });
      }
    },
  };
}

export default defineConfig({
  plugins: [react(), copyStaticAssets()],
  worker: { format: "es" },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        popup: resolve(here, "src/popup/index.html"),
        options: resolve(here, "src/options/index.html"),
      },
    },
  },
});
