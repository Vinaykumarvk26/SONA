export default function ConfidenceMeter({ confidence }) {
  const percent = Math.max(0, Math.min(100, Math.round((confidence || 0) * 100)));

  return (
    <div>
      <div className="conf-bar">
        <div
          className="h-2 rounded-full bg-gradient-to-r from-accent to-success transition-all duration-300"
          style={{ width: `${percent}%` }}
        />
      </div>
      <p className="mt-2 text-xs text-text1">Confidence: {percent}%</p>
    </div>
  );
}
