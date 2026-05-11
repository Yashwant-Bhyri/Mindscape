const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8002/api";

export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    cache: "no-store",
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const payload = await response.json();
      detail = payload.detail ?? detail;
    } catch {}
    throw new Error(detail);
  }

  return response.json() as Promise<T>;
}

export async function postJson<T>(path: string, body: Record<string, unknown>): Promise<T> {
  return api<T>(path, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function postFormData<T>(path: string, formData: FormData): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    body: formData,
    cache: "no-store",
  });

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const payload = await response.json();
      detail = payload.detail ?? detail;
    } catch {}
    throw new Error(detail);
  }

  return response.json() as Promise<T>;
}

export function appPath(path: string) {
  return path.startsWith("/") ? path : `/${path}`;
}
