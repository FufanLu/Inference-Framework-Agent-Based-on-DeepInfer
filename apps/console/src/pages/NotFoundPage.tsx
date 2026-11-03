import { Link } from "react-router";
import { PageHeader } from "@/components/PageHeader";
import { EmptyState } from "@/components/States";
import { useT } from "@/i18n";

export function NotFoundPage() {
  const t = useT();
  return (
    <div className="flex flex-col gap-6">
      <PageHeader title={t.app.notFoundTitle} />
      <EmptyState title={t.app.notFoundTitle}>
        <p>
          {t.app.notFoundBody}{" "}
          <Link to="/" className="text-ink underline">
            {t.app.notFoundLink}
          </Link>
          {t.app.notFoundAfter}
        </p>
      </EmptyState>
    </div>
  );
}
