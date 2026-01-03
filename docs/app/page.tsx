import Link from 'next/link';

export default function HomePage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8 bg-gradient-to-b from-slate-900 to-slate-800 text-white">
      <div className="max-w-4xl text-center">
        <h1 className="text-5xl font-bold mb-4">lux-fhe</h1>
        <p className="text-xl text-slate-300 mb-8">
          GPU-accelerated Fully Homomorphic Encryption library for privacy-preserving computation
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <div className="p-6 bg-slate-700/50 rounded-lg">
            <h3 className="text-lg font-semibold text-cyan-400 mb-2">TFHE</h3>
            <p className="text-sm text-slate-400">
              Fast bootstrapping for binary circuits with GPU acceleration
            </p>
          </div>
          <div className="p-6 bg-slate-700/50 rounded-lg">
            <h3 className="text-lg font-semibold text-cyan-400 mb-2">BGV/BFV/CKKS</h3>
            <p className="text-sm text-slate-400">
              Leveled HE schemes for arithmetic operations on encrypted data
            </p>
          </div>
          <div className="p-6 bg-slate-700/50 rounded-lg">
            <h3 className="text-lg font-semibold text-cyan-400 mb-2">Threshold FHE</h3>
            <p className="text-sm text-slate-400">
              Multi-party computation with distributed key generation
            </p>
          </div>
        </div>

        <div className="flex gap-4 justify-center">
          <Link
            href="/docs"
            className="px-6 py-3 bg-cyan-600 hover:bg-cyan-500 rounded-lg font-medium transition-colors"
          >
            Get Started →
          </Link>
          <Link
            href="https://github.com/luxfi/fhe"
            className="px-6 py-3 bg-slate-700 hover:bg-slate-600 rounded-lg font-medium transition-colors"
          >
            GitHub
          </Link>
        </div>
      </div>
    </main>
  );
}
