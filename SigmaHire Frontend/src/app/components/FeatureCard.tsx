export default function FeatureCard({ title, icon, desc, bullets, color }) {
  return (
    <div className="bg-white shadow rounded-2xl p-6 w-full sm:w-80 text-gray-800 bg-white hover:bg-gray-100">
      <h3 className={`text-xl font-semibold mb-2 text-${color}`}>{title}</h3>
      <p className="text-gray-600 mb-3">{desc}</p>
      <ul className="text-sm text-gray-500 list-disc list-inside">
        {bullets.map((b, i) => (
          <li key={i}>{b}</li>
        ))}
      </ul>
    </div>
  );
}