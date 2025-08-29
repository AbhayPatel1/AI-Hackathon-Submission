// components/Footer.tsx
export default function Footer() {
  return (
    <footer className="w-full fixed bottom-0 left-0 px-6 py-3 flex items-center justify-between bg-[#1f1f1f] text-sm text-gray-400 z-50">
      <p>© {new Date().getFullYear()} Sigmoid. All rights reserved.</p>
      <p>Made by team <span className="text-white font-medium">Hidden Layers</span></p>
    </footer>
  );
}