import { QueryClient, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSettings } from "@/state/settings";
import { api, isGatewayError } from "./client";
import type { RouterSnapshot, RouterUpdate, TenantUpsert } from "./types";

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Auth and missing-resource errors do not heal by retrying.
      retry: (failureCount, error) => {
        if (isGatewayError(error) && error.status !== null && error.status < 500) return false;
        return failureCount < 1;
      },
      refetchOnWindowFocus: true,
      staleTime: 0,
    },
  },
});

/**
 * Poll every `ms`, but stop after a 401: the token will not start working by
 * itself, and changing it in Settings changes the query key and refetches.
 */
function every(ms: number | false) {
  return (query: { state: { error: unknown } }) => {
    const error = query.state.error;
    return isGatewayError(error) && error.unauthorized ? false : ms;
  };
}

/** Query keys carry the connection so that changing settings refetches. */
function useConnectionKey() {
  const { gatewayUrl, adminToken, apiKey } = useSettings();
  return [gatewayUrl, adminToken.length > 0 ? adminToken : null, apiKey.length > 0 ? apiKey : null] as const;
}

export function useGatewayHealth() {
  const connection = useConnectionKey();
  return useQuery({
    queryKey: ["healthz", ...connection],
    queryFn: ({ signal }) => api.healthz(signal),
    refetchInterval: every(5_000),
    retry: false,
  });
}

export function useOverview() {
  const connection = useConnectionKey();
  return useQuery({
    queryKey: ["overview", ...connection],
    queryFn: ({ signal }) => api.overview(signal),
    refetchInterval: every(5_000),
  });
}

/** Server-side KPIs over the last `windowS` seconds (GET /admin/stats). */
export function useStats(windowS: number, intervalMs = 2_000) {
  const connection = useConnectionKey();
  return useQuery({
    queryKey: ["stats", windowS, ...connection],
    queryFn: ({ signal }) => api.stats(windowS, signal),
    refetchInterval: every(intervalMs),
    placeholderData: (previous) => previous,
  });
}

export function useRouterSnapshot(intervalMs = 1_000) {
  const connection = useConnectionKey();
  return useQuery({
    queryKey: ["router", ...connection],
    queryFn: ({ signal }) => api.router(signal),
    refetchInterval: every(intervalMs),
  });
}

export function useWorkers(intervalMs = 5_000) {
  const connection = useConnectionKey();
  return useQuery({
    queryKey: ["workers", ...connection],
    queryFn: ({ signal }) => api.workers(signal),
    refetchInterval: every(intervalMs),
  });
}

export function useRequests(limit: number, intervalMs: number | false = 2_000) {
  const connection = useConnectionKey();
  return useQuery({
    queryKey: ["requests", limit, ...connection],
    queryFn: ({ signal }) => api.requests(limit, signal),
    refetchInterval: every(intervalMs),
    placeholderData: (previous) => previous,
  });
}

export function useRequestDetail(id: string | null) {
  const connection = useConnectionKey();
  return useQuery({
    queryKey: ["request", id, ...connection],
    queryFn: ({ signal }) => api.request(id as string, signal),
    enabled: id !== null,
    staleTime: Infinity,
  });
}

export function useModels() {
  const connection = useConnectionKey();
  return useQuery({
    queryKey: ["models", ...connection],
    queryFn: ({ signal }) => api.models(signal),
    refetchInterval: every(10_000),
  });
}

/** PUT /admin/router; the response is the new snapshot. */
export function useUpdateRouter() {
  const client = useQueryClient();
  const connection = useConnectionKey();
  return useMutation({
    mutationFn: (update: RouterUpdate) => api.updateRouter(update),
    onSuccess: (snapshot: RouterSnapshot) => {
      client.setQueryData(["router", ...connection], snapshot);
      void client.invalidateQueries({ queryKey: ["overview"] });
    },
  });
}

export function useAddWorker() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (endpoint: string) => api.addWorker(endpoint),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["workers"] });
      void client.invalidateQueries({ queryKey: ["router"] });
    },
  });
}

// ------------------------------------------------------------ control plane

export function useTenants() {
  const connection = useConnectionKey();
  return useQuery({
    queryKey: ["tenants", ...connection],
    queryFn: ({ signal }) => api.tenants(signal),
    refetchInterval: every(10_000),
  });
}

export function useKeys(tenant: string | null) {
  const connection = useConnectionKey();
  return useQuery({
    queryKey: ["keys", tenant, ...connection],
    queryFn: ({ signal }) => api.keys(tenant, signal),
    enabled: tenant !== null,
  });
}

export function useUsage(tenant: string | null, days: number) {
  const connection = useConnectionKey();
  return useQuery({
    queryKey: ["usage", tenant, days, ...connection],
    queryFn: ({ signal }) => api.usage(tenant, days, signal),
    refetchInterval: every(15_000),
    placeholderData: (previous) => previous,
  });
}

export function useUpsertTenant() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (tenant: TenantUpsert) => api.upsertTenant(tenant),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["tenants"] });
    },
  });
}

export function useCreateKey() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ tenant, name }: { tenant: string; name: string }) => api.createKey(tenant, name),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["keys"] });
    },
  });
}

export function useRevokeKey() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.revokeKey(id),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["keys"] });
    },
  });
}
