import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClientProvider } from "@tanstack/react-query";
import { createBrowserRouter, RouterProvider } from "react-router";
import { queryClient } from "@/api/queries";
import { applyLocale } from "@/i18n";
import { applyTheme } from "@/state/settings";
import { App } from "./App";
import { OverviewPage } from "./pages/OverviewPage";
import { NotFoundPage } from "./pages/NotFoundPage";
import "./index.css";

applyTheme();
applyLocale();

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    // Shown for the moment a lazily loaded page is fetched on first load.
    hydrateFallbackElement: <div className="min-h-dvh bg-paper" aria-busy="true" />,
    children: [
      { index: true, element: <OverviewPage /> },
      // Other pages load on first visit.
      { path: "cluster", lazy: async () => ({ Component: (await import("./pages/ClusterPage")).ClusterPage }) },
      { path: "playground", lazy: async () => ({ Component: (await import("./pages/PlaygroundPage")).PlaygroundPage }) },
      { path: "requests", lazy: async () => ({ Component: (await import("./pages/RequestsPage")).RequestsPage }) },
      { path: "agent", lazy: async () => ({ Component: (await import("./pages/AgentStudioPage")).AgentStudioPage }) },
      { path: "tenants", lazy: async () => ({ Component: (await import("./pages/TenantsPage")).TenantsPage }) },
      { path: "*", element: <NotFoundPage /> },
    ],
  },
]);

const root = document.getElementById("root");
if (!root) throw new Error("missing #root");

createRoot(root).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
);
