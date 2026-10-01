import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "C++ Systems & 22 LLM CUDA Kernels Mastery — Digital Book & Lab",
  description: "Interactive learning platform for C++ memory systems, CUDA GPU programming, and the 22 production LLM CUDA kernels.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark h-full" suppressHydrationWarning>
      <body
        className="h-full bg-[#080b11] text-slate-200 antialiased overflow-hidden"
        suppressHydrationWarning
      >
        {children}
      </body>
    </html>
  );
}
