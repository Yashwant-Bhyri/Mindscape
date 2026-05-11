import type { Metadata } from "next";

import "./globals.css";
import "./redesign.css";

export const metadata: Metadata = {
  title: "MindScape Clinical OS",
  description: "Next-generation psychiatric care workflow over a Python clinical engine",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
