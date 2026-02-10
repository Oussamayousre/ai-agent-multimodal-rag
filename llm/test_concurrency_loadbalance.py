# test_concurrency_loadbalance.py
import asyncio
import httpx
import time
from collections import Counter

async def send_request(client, prompt, request_id):
    url = "http://127.0.0.1:8265/qwen" # Default Ray Serve address and route_prefix
    headers = {"Content-Type": "application/json"}
    payload = {"prompt": prompt}
    start_time = time.time()
    try:
        response = await client.post(url, headers=headers, json=payload, timeout=300) # Increased timeout
        response.raise_for_status()
        data = response.json()
        end_time = time.time()
        print(f"Request {request_id} completed by PID {data.get('replica_pid')} in {end_time - start_time:.2f}s")
        return {
            "request_id": request_id,
            "replica_pid": data.get("replica_pid"),
            "generated_text": data.get("generated_text"),
            "latency": end_time - start_time,
            "status": "success"
        }
    except httpx.RequestError as e:
        end_time = time.time()
        print(f"Request {request_id} failed: {e} in {end_time - start_time:.2f}s")
        return {
            "request_id": request_id,
            "replica_pid": None,
            "generated_text": None,
            "latency": end_time - start_time,
            "status": "failure",
            "error": str(e)
        }
    except httpx.HTTPStatusError as e:
        end_time = time.time()
        print(f"Request {request_id} failed with HTTP error {e.response.status_code}: {e.response.text} in {end_time - start_time:.2f}s")
        return {
            "request_id": request_id,
            "replica_pid": None,
            "generated_text": None,
            "latency": end_time - start_time,
            "status": "failure",
            "error": f"HTTP {e.response.status_code}: {e.response.text}"
        }


async def main():
    num_concurrent_requests = 10
    prompts = [f"Tell me a short story about a robot, request {i}." for i in range(num_concurrent_requests)]
    
    print(f"Sending {num_concurrent_requests} concurrent requests...")
    
    async with httpx.AsyncClient() as client:
        tasks = [send_request(client, prompt, i) for i, prompt in enumerate(prompts)]
        results = await asyncio.gather(*tasks)

    successful_results = [r for r in results if r["status"] == "success"]
    failed_results = [r for r in results if r["status"] == "failure"]

    print("\n--- Test Results ---")
    print(f"Total requests sent: {len(results)}")
    print(f"Successful requests: {len(successful_results)}")
    print(f"Failed requests: {len(failed_results)}")

    if successful_results:
        # Concurrency check: all requests should ideally be processed in parallel
        total_latency = sum(r["latency"] for r in successful_results)
        avg_latency = total_latency / len(successful_results)
        print(f"Average successful request latency: {avg_latency:.2f}s")

        # Load balancing check
        pids = [r["replica_pid"] for r in successful_results if r["replica_pid"]]
        if pids:
            pid_counts = Counter(pids)
            print("\nReplica PID distribution:")
            for pid, count in pid_counts.items():
                print(f"  PID {pid}: {count} requests")
            
            if len(pid_counts) > 1:
                print("\nLoad balancing appears to be working across multiple replicas.")
            else:
                print("\nOnly one replica processed requests. Load balancing might not be effective or only one replica was active.")
        else:
            print("\nCould not determine replica PIDs for load balancing analysis.")
    
    if failed_results:
        print("\n--- Failed Requests Details ---")
        for fail in failed_results:
            print(f"Request {fail['request_id']}: {fail['error']} (Latency: {fail['latency']:.2f}s)")

if __name__ == "__main__":
    asyncio.run(main())