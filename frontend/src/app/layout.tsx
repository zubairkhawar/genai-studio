import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/contexts/ThemeContext";
import { GeneratingProvider } from "@/contexts/GeneratingContext";
import { AppShell } from "@/components/AppShell";
import { GlobalGeneratingModal } from "@/components/GlobalGeneratingModal";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "GenStudio",
  description: "AI-powered text-to-video, text-to-audio, and text-to-image generation platform",
  icons: {
    icon: '/logo.png',
    shortcut: '/logo.png',
    apple: '/logo.png',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <ThemeProvider>
          <GeneratingProvider>
            <AppShell>
              {children}
            </AppShell>
            <GlobalGeneratingModal />
          </GeneratingProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
