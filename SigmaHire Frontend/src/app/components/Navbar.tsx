'use client';
import { useRouter } from 'next/navigation';
import Image from 'next/image';

type NavbarProps = {
  showBackButton?: boolean;
  showActionButton?: boolean;
  actionLabel?: string;
  actionHref?: string;
};

export default function Navbar({
  showBackButton = false,
  showActionButton = false,
  actionLabel = '',
  actionHref = '',
}: NavbarProps) {
  const router = useRouter();

  return (
    <header className="w-full fixed top-0 left-0 bg-white border-b border-gray-200 z-50 px-6 py-4 flex items-center justify-between shadow-sm">
      <div className="flex items-center gap-4">
        {showBackButton ? (
          <button
            onClick={() => router.back()}
            className="text-sm text-gray-600 hover:text-[#7B61FF] font-medium"
          >
            ← Go Back
          </button>
        ) : (
          <Image src="/logo.png" alt="Sigmoid" width={120} height={32} />
        )}
      </div>

      {showActionButton && actionLabel && actionHref && (
        <a
          href={actionHref}
          target="_self"
          className="px-4 py-2 text-sm text-white bg-[#7B61FF] rounded-lg font-medium hover:bg-[#654fe0] transition"
        >
          {actionLabel}
        </a>
      )}
    </header>
  );
}