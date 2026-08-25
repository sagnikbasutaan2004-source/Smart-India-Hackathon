from motor.motor_asyncio import AsyncIOMotorClient
from backend.config import settings

client: AsyncIOMotorClient = None
db = None

async def connect_to_mongo():
    global client, db
    if not settings.MONGODB_URI:
        print("WARNING: MONGODB_URI not set.")
        return
    client = AsyncIOMotorClient(settings.MONGODB_URI)
    db = client[settings.MONGODB_DATABASE]
    print(f"[MongoDB] Connected -> database: {settings.MONGODB_DATABASE}")

async def close_mongo_connection():
    global client
    if client:
        client.close()
        print("[MongoDB] Connection closed.")

def get_db():
    return db
