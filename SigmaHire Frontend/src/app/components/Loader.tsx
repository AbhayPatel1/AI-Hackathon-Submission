// components/Loader.tsx
import Image from 'next/image';

export default function Loader() {
  return (
    <div className="fixed inset-0 bg-white flex flex-col items-center justify-center z-[9999] transition-opacity">
      <Image src="/logo.png" alt="Sigmoid" width={100} height={40} />
      <div className="mt-4 w-6 h-6 border-4 border-[#7B61FF] border-t-transparent rounded-full animate-spin" />
    </div>
  );
}