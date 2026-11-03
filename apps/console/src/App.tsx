import { useEffect } from "react";
import { Outlet, useLocation } from "react-router";
import { liveHub } from "@/api/live";
import { Sidebar } from "@/components/Sidebar";
import { SettingsDialog } from "@/components/SettingsDialog";
import { useT } from "@/i18n";

const PAGES = {
  "/": "overview",
  "/cluster": "cluster",
  "/playground": "playground",
  "/requests": "requests",
  "/agent": "agent",
  "/tenants": "tenants",
} as const;

export function App() {
  const location = useLocation();
  const t = useT();

  useEffect(() => {
    liveHub.start();
  }, []);

  useEffect(() => {
    const key = PAGES[location.pathname as keyof typeof PAGES];
    document.title = key ? `${t.nav[key]} · ${t.app.documentTitle}` : t.app.documentTitle;
  }, [location.pathname, t]);

  return (
    <div className="flex min-h-dvh bg-paper text-ink">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 focus:bg-ink focus:px-3 focus:py-2 focus:text-paper"
      >
        {t.app.skip}
      </a>
      <Sidebar />
      <main id="main" className="min-w-0 flex-1 px-6 pt-5 pb-12 xl:px-8">
        <Outlet />
      </main>
      <SettingsDialog />
    </div>
  );
}
