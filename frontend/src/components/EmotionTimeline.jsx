import {
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LineElement,
  LinearScale,
  PointElement,
  Tooltip,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend);

export default function EmotionTimeline({ points }) {
  const labels = points.map((p) => new Date(p.timestamp).toLocaleTimeString());
  const values = points.map((p) => Number((p.confidence || 0).toFixed(2)));

  const data = {
    labels,
    datasets: [
      {
        label: "Confidence",
        data: values,
        borderColor: "#22d3ee",
        backgroundColor: "rgba(34, 211, 238, 0.2)",
        fill: true,
        tension: 0.28,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: "#cbd5e1" },
      },
    },
    scales: {
      y: {
        min: 0,
        max: 1,
        ticks: { color: "#94a3b8" },
        grid: { color: "rgba(148, 163, 184, 0.2)" },
      },
      x: {
        ticks: { color: "#94a3b8" },
        grid: { color: "rgba(148, 163, 184, 0.1)" },
      },
    },
  };

  return (
    <div className="glass p-5 h-72 animate-floatIn">
      <h3 className="font-heading text-lg text-text0">Emotion Timeline</h3>
      <div className="mt-4 h-52">
        <Line data={data} options={options} />
      </div>
    </div>
  );
}
