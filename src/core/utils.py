import asyncio
import concurrent.futures

def run_sync(coro_func, *args, **kwargs):
    """
    Safely execute an asynchronous coroutine function in an isolated thread context.
    We pass the function and bounds, NOT an instantiated coroutine object, 
    to strictly prevent ContextVar collisions inheriting from Jupyter's main thread loop.
    """
    def _run():
        return asyncio.run(coro_func(*args, **kwargs))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(_run).result()
