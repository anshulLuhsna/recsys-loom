import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Loom Atelier",
  description: "Personalized H&M recommendations and catalog search",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <nav className="nav">
            <Link className="brand" href="/">
              Loom Atelier
            </Link>
            <div>
              <Link href="/">For You</Link>
              <Link href="/search">Search</Link>
            </div>
          </nav>
          {children}
        </div>
      </body>
    </html>
  );
}
