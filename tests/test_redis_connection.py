"""
Redis Connection Test
Tests Redis connectivity with your configured URL (Upstash or ElastiCache).
"""

import os
import time
from dotenv import load_dotenv
load_dotenv()

try:
    import redis
except ImportError:
    print("❌ ERROR: redis library not installed")
    print("\nInstall with:")
    print("  pip install redis")
    exit(1)

redis_url = os.getenv("REDIS_URL")

print("=" * 70)
print("REDIS CONNECTION TEST")
print("=" * 70)

if not redis_url:
    print("❌ ERROR: REDIS_URL not set in .env file")
    print("\nAdd to .env file:")
    print("  REDIS_URL=redis://your-redis-host:6379")
    print("  Or for Upstash: REDIS_URL=https://your-upstash.io")
    exit(1)

# Mask password in URL for display
display_url = redis_url
if '@' in redis_url:
    parts = redis_url.split('@')
    display_url = f"{parts[0].split('://')[0]}://***@{parts[1]}"

print(f"Testing connection to: {display_url}")
print()

try:
    # Create Redis client
    print("Test 1: Creating Redis client...")
    
    # Handle Upstash HTTPS URLs
    if redis_url.startswith("https://"):
        # Upstash Redis with REST API
        redis_token = os.getenv("REDIS_TOKEN")
        if not redis_token:
            print("⚠️  WARNING: Upstash HTTPS detected but REDIS_TOKEN not set")
            print("   This may fail. Check your .env file.")
        
        # Try standard redis connection (Upstash supports Redis protocol too)
        # Convert https:// to rediss:// for SSL
        redis_url_ssl = redis_url.replace("https://", "rediss://")
        client = redis.from_url(
            redis_url_ssl,
            socket_timeout=5,
            socket_connect_timeout=5,
            decode_responses=True
        )
    else:
        client = redis.from_url(
            redis_url,
            socket_timeout=5,
            socket_connect_timeout=5,
            decode_responses=True
        )
    
    print("✅ Redis client created")
    
    # Test 2: PING
    print("\nTest 2: Sending PING command...")
    response = client.ping()
    print(f"✅ PING successful! Response: {response}")
    
    # Test 3: SET
    print("\nTest 3: Writing test data (SET)...")
    test_key = "valuenaire:test:connection"
    test_value = f"Connection test successful at {time.strftime('%Y-%m-%d %H:%M:%S')}"
    client.set(test_key, test_value, ex=60)  # Expires in 60 seconds
    print(f"✅ SET successful!")
    print(f"   Key: {test_key}")
    print(f"   Value: {test_value}")
    
    # Test 4: GET
    print("\nTest 4: Reading test data (GET)...")
    retrieved = client.get(test_key)
    print(f"✅ GET successful!")
    print(f"   Retrieved: {retrieved}")
    
    # Test 5: EXISTS
    print("\nTest 5: Checking key existence (EXISTS)...")
    exists = client.exists(test_key)
    print(f"✅ EXISTS check passed! Key exists: {bool(exists)}")
    
    # Test 6: TTL
    print("\nTest 6: Checking TTL...")
    ttl = client.ttl(test_key)
    print(f"✅ TTL check passed! Expires in: {ttl} seconds")
    
    # Test 7: INCR (test counter operations)
    print("\nTest 7: Testing counter operations (INCR)...")
    counter_key = "valuenaire:test:counter"
    count = client.incr(counter_key)
    print(f"✅ INCR successful! Counter: {count}")
    
    # Test 8: DEL
    print("\nTest 8: Cleaning up test data (DEL)...")
    deleted = client.delete(test_key, counter_key)
    print(f"✅ DEL successful! Deleted {deleted} keys")
    
    # Test 9: INFO (get Redis server info)
    print("\nTest 9: Getting Redis server info...")
    try:
        info = client.info("server")
        print(f"✅ INFO successful!")
        print(f"   Redis version: {info.get('redis_version', 'N/A')}")
        print(f"   OS: {info.get('os', 'N/A')}")
        print(f"   Uptime: {info.get('uptime_in_days', 'N/A')} days")
    except Exception as e:
        print(f"⚠️  INFO command restricted (common on Upstash): {e}")
    
    print("\n" + "=" * 70)
    print("🎉 ALL TESTS PASSED - REDIS IS READY!")
    print("=" * 70)
    print("\nYour application can now:")
    print("  ✅ Use Redis for L1/L2 caching")
    print("  ✅ Store semantic cache embeddings")
    print("  ✅ Share cache across multiple instances")
    print("\nRedis features enabled:")
    print("  ✅ HotPromptCache L2 (distributed)")
    print("  ✅ Schema metadata caching")
    print("  ✅ Session state management")

except redis.exceptions.ConnectionError as e:
    print(f"\n❌ CONNECTION ERROR: {e}")
    print("\n" + "=" * 70)
    print("TROUBLESHOOTING")
    print("=" * 70)
    print("\nPossible causes:")
    print("  1. Redis server is not running")
    print("  2. Incorrect host/port in REDIS_URL")
    print("  3. Firewall blocking connection")
    print("  4. ElastiCache security group not configured")
    print("\nSolutions:")
    print("  1. Verify REDIS_URL format:")
    print("     - Standard: redis://host:6379")
    print("     - With auth: redis://:password@host:6379")
    print("     - SSL: rediss://host:6379")
    print("     - Upstash: https://your-instance.upstash.io")
    print("\n  2. For ElastiCache:")
    print("     - Check security group allows port 6379")
    print("     - Verify your IP is whitelisted")
    print("     - Must be in same VPC or use VPN")
    print("\n  3. For Upstash:")
    print("     - Verify REDIS_TOKEN is set (if using HTTPS)")
    print("     - Check endpoint URL is correct")
    exit(1)

except redis.exceptions.AuthenticationError as e:
    print(f"\n❌ AUTHENTICATION ERROR: {e}")
    print("\n" + "=" * 70)
    print("TROUBLESHOOTING")
    print("=" * 70)
    print("\nPossible causes:")
    print("  1. Incorrect password in REDIS_URL")
    print("  2. Redis server requires AUTH but none provided")
    print("\nSolutions:")
    print("  1. Check REDIS_URL format with password:")
    print("     redis://:your_password@host:6379")
    print("\n  2. For Upstash, set REDIS_TOKEN in .env")
    exit(1)

except redis.exceptions.TimeoutError as e:
    print(f"\n❌ TIMEOUT ERROR: {e}")
    print("\n" + "=" * 70)
    print("TROUBLESHOOTING")
    print("=" * 70)
    print("\nPossible causes:")
    print("  1. Network latency too high")
    print("  2. Redis server overloaded")
    print("  3. Firewall dropping packets")
    print("\nSolutions:")
    print("  1. Check network connectivity to Redis server")
    print("  2. Try increasing socket_timeout in code")
    print("  3. Verify Redis server health")
    exit(1)

except Exception as e:
    print(f"\n❌ UNEXPECTED ERROR: {e}")
    print("\nError type:", type(e).__name__)
    
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
    exit(1)
