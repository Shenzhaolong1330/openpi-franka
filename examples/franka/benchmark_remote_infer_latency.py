#!/usr/bin/env python3
"""Benchmark remote inference latency via WebSocket.

This script tests the inference latency of a remote policy server.

Usage:
    python examples/franka/benchmark_remote_infer_latency.py --host 100.101.84.9 --port 8000
"""

import argparse
import time
import statistics
import socket
import numpy as np

import requests


def check_http_connection(host: str, port: int, timeout: int = 5) -> bool:
    """Check if we can reach the server via HTTP."""
    try:
        url = f"http://{host}:{port}/healthz"
        resp = requests.get(url, timeout=timeout)
        print(f"  HTTP health check: {resp.status_code} - {resp.text.strip()}")
        return resp.status_code == 200
    except requests.exceptions.ConnectionError as e:
        print(f"  HTTP connection failed: {e}")
        return False
    except requests.exceptions.Timeout:
        print(f"  HTTP connection timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"  HTTP check error: {e}")
        return False


def check_tcp_connection(host: str, port: int, timeout: int = 5) -> bool:
    """Check if we can reach the server via TCP."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        if result == 0:
            print(f"  TCP port {port} is OPEN")
            return True
        else:
            print(f"  TCP port {port} is CLOSED (error code: {result})")
            return False
    except socket.timeout:
        print(f"  TCP connection timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"  TCP check error: {e}")
        return False


def make_dummy_observation(action_mode: str = "delta_ee") -> dict:
    """Create a dummy observation for testing."""
    if action_mode == "delta_ee":
        state = np.random.rand(7).astype(np.float32)
    else:
        state = np.random.rand(8).astype(np.float32)
    
    return {
        "observation/state": state,
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the object and place it in the basket"
    }


def benchmark_remote_inference(client, num_warmup: int = 3, num_iterations: int = 50, action_mode: str = "delta_ee"):
    """Benchmark remote inference latency.
    
    Args:
        client: WebSocket client policy
        num_warmup: Number of warmup iterations
        num_iterations: Number of iterations to measure
        action_mode: "delta_ee" or "joint"
    
    Returns:
        Dictionary with latency statistics
    """
    latencies = []
    server_infer_times = []
    
    print(f"\n{'='*60}")
    print(f"Remote Inference Latency Benchmark")
    print(f"{'='*60}")
    print(f"Action mode: {action_mode}")
    print(f"Warmup iterations: {num_warmup}")
    print(f"Measurement iterations: {num_iterations}")
    print(f"{'='*60}\n")
    
    # Warmup
    print("Running warmup...")
    obs = make_dummy_observation(action_mode)
    for i in range(num_warmup):
        _ = client.infer(obs)
    
    print("Starting benchmark...")
    
    for i in range(num_iterations):
        obs = make_dummy_observation(action_mode)
        
        start = time.perf_counter()
        result = client.infer(obs)
        end = time.perf_counter()
        
        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)
        
        # Extract server-side timing if available
        if "server_timing" in result:
            server_infer_times.append(result["server_timing"].get("infer_ms", 0))
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{num_iterations} iterations")
    
    # Calculate statistics
    stats = {
        "mean_ms": statistics.mean(latencies),
        "std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "fps": 1000 / statistics.mean(latencies),
    }
    
    if server_infer_times:
        stats["server_mean_ms"] = statistics.mean(server_infer_times)
        stats["server_std_ms"] = statistics.stdev(server_infer_times) if len(server_infer_times) > 1 else 0
        stats["network_overhead_ms"] = stats["mean_ms"] - stats["server_mean_ms"]
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Benchmark remote OpenPI inference latency")
    parser.add_argument("--host", type=str, default="100.101.84.9", help="Server host IP")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--action-mode", type=str, default="delta_ee", choices=["delta_ee", "joint"],
                        help="Action mode to test")
    parser.add_argument("--api-key", type=str, default=None, help="API key for authentication")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("Connection Diagnostics")
    print(f"{'='*60}")
    print(f"Target: {args.host}:{args.port}")
    
    # Run diagnostics
    print("\n1. Checking TCP connection...")
    tcp_ok = check_tcp_connection(args.host, args.port)
    
    print("\n2. Checking HTTP health endpoint...")
    http_ok = check_http_connection(args.host, args.port)
    
    if not tcp_ok:
        print("\n❌ TCP connection failed. Server may not be running or firewall is blocking.")
        print("   Please check:")
        print(f"   - Is the server running on {args.host}:{args.port}?")
        print("   - Is there a firewall blocking the connection?")
        print("   - Can you ping the server?")
        return
    
    if not http_ok:
        print("\n⚠️  HTTP health check failed, but TCP port is open.")
        print("   The server might be running but not responding to HTTP requests.")
    
    print("\n3. Attempting WebSocket connection...")
    
    # Import after argparse to avoid slow imports for --help
    from openpi_client import websocket_client_policy
    
    try:
        client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host, 
            port=args.port,
            api_key=args.api_key
        )
        print(f"   ✓ WebSocket connected successfully!")
        print(f"   Server metadata: {client.get_server_metadata()}")
    except Exception as e:
        print(f"   ❌ WebSocket connection failed: {e}")
        print("\nPossible causes:")
        print("   - Server is not running")
        print("   - Server is running but crashed during startup")
        print("   - WebSocket protocol mismatch")
        print("   - Network/firewall issues")
        return
    
    # Run benchmark
    stats = benchmark_remote_inference(
        client,
        num_warmup=args.warmup,
        num_iterations=args.iterations,
        action_mode=args.action_mode
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Total latency (client-side):")
    print(f"    Mean:          {stats['mean_ms']:.2f} ms")
    print(f"    Std deviation: {stats['std_ms']:.2f} ms")
    print(f"    Min:           {stats['min_ms']:.2f} ms")
    print(f"    Max:           {stats['max_ms']:.2f} ms")
    print(f"    Median:        {stats['median_ms']:.2f} ms")
    print(f"    P95:           {stats['p95_ms']:.2f} ms")
    print(f"    P99:           {stats['p99_ms']:.2f} ms")
    print(f"    Throughput:    {stats['fps']:.1f} FPS")
    
    if "server_mean_ms" in stats:
        print(f"\n  Server-side inference time:")
        print(f"    Mean:          {stats['server_mean_ms']:.2f} ms")
        print(f"    Std deviation: {stats['server_std_ms']:.2f} ms")
        print(f"\n  Network overhead:")
        print(f"    Mean:          {stats['network_overhead_ms']:.2f} ms")
        print(f"    Percentage:    {stats['network_overhead_ms']/stats['mean_ms']*100:.1f}%")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
