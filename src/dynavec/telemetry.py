import json
import time
import random
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np

from dynavec.cache import SemanticCache
from dynavec.utils import timed

# Standalone React Dashboard embedded inside the Python module
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dynavec Telemetry</title>
    <!-- Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <!-- React & ReactDOM -->
    <script crossorigin src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <!-- PropTypes (Required by Recharts) -->
    <script crossorigin src="https://unpkg.com/prop-types@15.8.1/prop-types.min.js"></script>
    <!-- Recharts -->
    <script crossorigin src="https://unpkg.com/recharts@2.1.9/umd/Recharts.js"></script>
    <!-- Babel for in-browser JSX compilation -->
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <style>
        :root {
            --bg-dark: #0f0f13;
            --bg-card: #181820;
            --bg-hover: #22222d;
            --border-color: #2a2a35;
            --text-primary: #f8f8f8;
            --text-secondary: #8b8b9b;
            --accent-blue: #3b82f6;
            --accent-green: #10b981;
            --accent-yellow: #f59e0b;
            --accent-red: #ef4444;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.5);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.5), 0 2px 4px -1px rgba(0, 0, 0, 0.3);
        }
        body {
            margin: 0;
            background-color: var(--bg-dark);
            color: var(--text-primary);
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            -webkit-font-smoothing: antialiased;
        }
        .dashboard-container { max-width: 1280px; margin: 0 auto; padding: 2rem; }
        .dashboard-header {
            display: flex; justify-content: space-between; align-items: flex-end;
            margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 1px solid var(--border-color);
        }
        .dashboard-title {
            font-size: 1.875rem; font-weight: 700; margin: 0 0 0.25rem 0;
            background: linear-gradient(135deg, #fff, #8b8b9b);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
        }
        .dashboard-subtitle { color: var(--text-secondary); margin: 0; font-size: 0.95rem; }
        .range-selector { display: flex; align-items: center; gap: 1rem; }
        .range-label { color: var(--text-secondary); font-size: 0.875rem; font-weight: 500; }
        .range-buttons {
            display: flex; background: #111115; border: 1px solid var(--border-color);
            border-radius: 8px; padding: 0.25rem; box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
        }
        .range-btn {
            background: transparent; border: none; color: var(--text-secondary);
            padding: 0.375rem 1rem; font-size: 0.875rem; font-weight: 500;
            border-radius: 6px; cursor: pointer; transition: all 0.2s ease;
        }
        .range-btn:hover { color: var(--text-primary); }
        .range-btn.active { background: var(--bg-hover); color: var(--text-primary); box-shadow: var(--shadow-sm); }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }
        .metric-card {
            background: var(--bg-card); border: 1px solid var(--border-color);
            border-radius: 16px; padding: 1.5rem; box-shadow: var(--shadow-sm);
            position: relative; overflow: hidden; transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .metric-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); border-color: #3b3b4b; }
        .metric-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
        .metric-title { color: var(--text-secondary); font-size: 0.875rem; font-weight: 500; margin: 0; }
        .metric-value { font-size: 2.25rem; font-weight: 700; margin-bottom: 0.5rem; color: var(--text-primary); display: flex; align-items: baseline; gap: 0.25rem; }
        .text-sm { font-size: 1rem; color: var(--text-secondary); font-weight: 500; }
        .metric-trend { font-size: 0.875rem; font-weight: 500; color: var(--text-secondary); }
        .gauge-container { height: 6px; background: #2a2a35; border-radius: 3px; margin-top: 1rem; overflow: hidden; }
        .gauge-fill { height: 100%; background: linear-gradient(90deg, var(--accent-blue), var(--accent-green)); border-radius: 3px; transition: width 0.5s ease-out; }
        .charts-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 1.5rem; }
        @media (max-width: 1024px) { .charts-grid { grid-template-columns: 1fr; } }
        .chart-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 16px; padding: 1.5rem; box-shadow: var(--shadow-sm); }
        .chart-title { font-size: 1.125rem; font-weight: 600; margin: 0 0 1.5rem 0; color: var(--text-primary); }
        .chart-wrapper { height: 350px; width: 100%; }
        .loading { display: flex; justify-content: center; align-items: center; height: 100vh; font-size: 1.25rem; color: var(--text-secondary); }
    </style>
</head>
<body>
    <div id="root"></div>
    <script type="text/babel">
        const { useState, useEffect } = React;
        const { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } = window.Recharts;

        function TelemetryDashboard() {
            const [timeRange, setTimeRange] = useState(1);
            const [data, setData] = useState([]);
            const [error, setError] = useState(null);
            
            useEffect(() => {
                setData([]);
                const fetchMetrics = async () => {
                    try {
                        const res = await fetch('/metrics');
                        if (!res.ok) throw new Error('API down');
                        const metrics = await res.json();
                        
                        setData(prev => {
                            const newTime = new Date();
                            const newDataPoint = {
                                time: newTime.toISOString(),
                                displayTime: newTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
                                p50: metrics.p50,
                                p95: metrics.p95,
                                p99: metrics.p99,
                                qpm: metrics.qpm,
                                cacheHitRate: metrics.cacheHitRate
                            };
                            const maxPoints = timeRange * 60 * 60;
                            const updated = [...prev, newDataPoint];
                            return updated.length > maxPoints ? updated.slice(-maxPoints) : updated;
                        });
                        setError(null);
                    } catch (err) {
                        console.error("Fetch error:", err);
                        setError("Unable to connect.");
                    }
                };
                
                fetchMetrics();
                const interval = setInterval(fetchMetrics, 1000);
                return () => clearInterval(interval);
            }, [timeRange]);

            const currentStats = data.length > 0 ? data[data.length - 1] : null;

            if (!currentStats) return <div className="loading">Loading telemetry...</div>;

            return (
                <div className="dashboard-container">
                    <header className="dashboard-header">
                        <div>
                            <h1 className="dashboard-title">Dynavec Telemetry</h1>
                            <p className="dashboard-subtitle">Real-time observability dashboard</p>
                        </div>
                        <div className="range-selector">
                            <span className="range-label">Time Range:</span>
                            <div className="range-buttons">
                                {[1, 6, 24].map(hours => (
                                    <button 
                                        key={hours}
                                        className={`range-btn ${timeRange === hours ? 'active' : ''}`}
                                        onClick={() => setTimeRange(hours)}
                                    >
                                        {hours}h
                                    </button>
                                ))}
                            </div>
                        </div>
                    </header>

                    <div className="metrics-grid">
                        <div className="metric-card">
                            <div className="metric-header">
                                <h3 className="metric-title">Queries / Min</h3>
                            </div>
                            <div className="metric-value">{currentStats.qpm.toLocaleString()}</div>
                            <div className="metric-trend">Live Traffic</div>
                        </div>
                        <div className="metric-card">
                            <div className="metric-header">
                                <h3 className="metric-title">Cache Hit Rate</h3>
                            </div>
                            <div className="metric-value">{currentStats.cacheHitRate.toFixed(1)}%</div>
                            <div className="metric-trend">SemanticCache</div>
                            <div className="gauge-container">
                                <div className="gauge-fill" style={{ width: `${currentStats.cacheHitRate}%` }}></div>
                            </div>
                        </div>
                        <div className="metric-card">
                            <div className="metric-header">
                                <h3 className="metric-title">p95 Latency</h3>
                            </div>
                            <div className="metric-value">{currentStats.p95} <span className="text-sm">ms</span></div>
                            <div className="metric-trend">End-to-end</div>
                        </div>
                    </div>

                    <div className="charts-grid">
                        <div className="chart-card large">
                            <h3 className="chart-title">Latency Percentiles (ms)</h3>
                            <div className="chart-wrapper">
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#2a2a35" vertical={false} />
                                        <XAxis dataKey="displayTime" stroke="#8b8b9b" tick={{ fill: '#8b8b9b', fontSize: 12 }} minTickGap={30} />
                                        <YAxis stroke="#8b8b9b" tick={{ fill: '#8b8b9b', fontSize: 12 }} />
                                        <Tooltip contentStyle={{ backgroundColor: '#1a1a24', borderColor: '#2a2a35', borderRadius: '8px', color: '#fff' }} />
                                        <Legend iconType="circle" wrapperStyle={{ paddingTop: '20px' }} />
                                        <Line type="monotone" dataKey="p50" name="p50" stroke="#10b981" strokeWidth={2} dot={false} isAnimationActive={false} />
                                        <Line type="monotone" dataKey="p95" name="p95" stroke="#f59e0b" strokeWidth={2} dot={false} isAnimationActive={false} />
                                        <Line type="monotone" dataKey="p99" name="p99" stroke="#ef4444" strokeWidth={2} dot={false} isAnimationActive={false} />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                        <div className="chart-card">
                            <h3 className="chart-title">Queries per Minute</h3>
                            <div className="chart-wrapper">
                                <ResponsiveContainer width="100%" height="100%">
                                    <AreaChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                                        <defs>
                                            <linearGradient id="colorQpm" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4}/>
                                                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#2a2a35" vertical={false} />
                                        <XAxis dataKey="displayTime" stroke="#8b8b9b" tick={{ fill: '#8b8b9b', fontSize: 12 }} minTickGap={30} />
                                        <YAxis stroke="#8b8b9b" tick={{ fill: '#8b8b9b', fontSize: 12 }} />
                                        <Tooltip contentStyle={{ backgroundColor: '#1a1a24', borderColor: '#2a2a35', borderRadius: '8px', color: '#fff' }} />
                                        <Area type="monotone" dataKey="qpm" name="QPM" stroke="#3b82f6" strokeWidth={2} fillOpacity={1} fill="url(#colorQpm)" isAnimationActive={false} />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    </div>
                </div>
            );
        }

        const root = ReactDOM.createRoot(document.getElementById('root'));
        root.render(<TelemetryDashboard />);
    </script>
</body>
</html>
"""

# Global state for telemetry
class TelemetryState:
    def __init__(self):
        self.latencies = []
        self.queries_count = 0
        self.start_time = time.time()
        self.cache = SemanticCache(threshold=0.90, max_size=100)

state = TelemetryState()

def latency_sink(name: str, duration: float):
    """Sink for dynavec @timed decorator."""
    state.latencies.append(duration * 1000) # ms
    state.queries_count += 1
    if len(state.latencies) > 1000:
        state.latencies.pop(0)

@timed(sink=latency_sink)
def simulated_search(query_vector):
    """Simulate a dynavec search that takes some time, hooked into @timed."""
    cached = state.cache.get("ns", query_vector, 10, {})
    if cached:
        time.sleep(random.uniform(0.005, 0.015)) # fast cache hit
        return cached
    
    time.sleep(random.uniform(0.040, 0.150))
    state.cache.put("ns", query_vector, 10, {}, [{"id": "doc1", "score": 1.0}])
    return None

def workload_thread():
    """Background thread generating real cache stats and latency data."""
    base_vector = np.random.rand(128).astype(np.float32)
    while True:
        if random.random() < 0.8:
            vec = base_vector + (np.random.rand(128).astype(np.float32) * 0.05)
        else:
            vec = np.random.rand(128).astype(np.float32)
        simulated_search(vec)
        time.sleep(random.uniform(0.01, 0.05))

class MetricsHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass # Suppress logging

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode('utf-8'))
            
        elif self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            lats = state.latencies[-100:] if state.latencies else [0]
            p50 = np.percentile(lats, 50) if lats else 0
            p95 = np.percentile(lats, 95) if lats else 0
            p99 = np.percentile(lats, 99) if lats else 0
            
            c_stats = state.cache.stats()
            elapsed = max(1, time.time() - state.start_time)
            qpm = (state.queries_count / elapsed) * 60
            
            metrics = {
                "p50": round(float(p50), 2),
                "p95": round(float(p95), 2),
                "p99": round(float(p99), 2),
                "qpm": round(qpm, 2),
                "cacheHitRate": round(c_stats["hit_rate"] * 100, 2),
                "cacheHits": c_stats["hits"],
                "cacheMisses": c_stats["misses"]
            }
            
            self.wfile.write(json.dumps(metrics).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def start_server(port=8778):
    """Start the telemetry dashboard server."""
    t = threading.Thread(target=workload_thread, daemon=True)
    t.start()
    
    server = HTTPServer(('', port), MetricsHandler)
    print(f"Dynavec Telemetry Dashboard running at: http://localhost:{port}/")
    print("Press Ctrl+C to quit.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down telemetry server.")
        server.server_close()
