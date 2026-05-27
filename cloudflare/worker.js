/**
 * Cloudflare Worker — proxy fapi.binance.com / api.bybit.com / data-api.binance.vision
 *
 * Motivação: GitHub Actions runners US recebem 451 (Binance) / 403 (Bybit) em endpoints
 * derivativos. Worker faz fetch a partir da rede CF (não-US em geral) → bypassa.
 *
 * Rotas:
 *   /binance-fapi/<path>?<query>  →  https://fapi.binance.com/<path>?<query>
 *   /binance-spot/<path>?<query>  →  https://data-api.binance.vision/<path>?<query>
 *   /bybit/<path>?<query>         →  https://api.bybit.com/<path>?<query>
 *
 * Segurança: PROXY_TOKEN no env do worker. Cliente envia header X-Proxy-Token.
 *   Se faltar/inválido → 401. Evita uso público abusivo.
 */

const ROUTES = {
  "/binance-fapi/": "https://fapi.binance.com/",
  "/binance-spot/": "https://data-api.binance.vision/",
  "/bybit/":        "https://api.bybit.com/",
};

export default {
  async fetch(request, env) {
    if (env.PROXY_TOKEN) {
      const got = request.headers.get("X-Proxy-Token");
      if (got !== env.PROXY_TOKEN) {
        return new Response("unauthorized", { status: 401 });
      }
    }

    const url = new URL(request.url);
    for (const [prefix, upstream] of Object.entries(ROUTES)) {
      if (url.pathname.startsWith(prefix)) {
        const rest = url.pathname.slice(prefix.length);
        const target = upstream + rest + url.search;
        const resp = await fetch(target, {
          method: request.method,
          headers: {
            "User-Agent": "btc-forecast-proxy/1.0",
            "Accept": "application/json",
          },
          // CF Worker fetch with cf options can hint to bypass cache and pick region
          cf: { cacheTtl: 0, cacheEverything: false },
        });
        const out = new Response(resp.body, resp);
        out.headers.set("X-Proxied-By", "btc-forecast-cf-worker");
        return out;
      }
    }
    return new Response("not found — use /binance-fapi/, /binance-spot/, /bybit/", { status: 404 });
  },
};
