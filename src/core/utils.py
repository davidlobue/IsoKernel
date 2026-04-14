import asyncio
import concurrent.futures

def run_sync(coro):
    """
    Safely execute an asynchronous coroutine in an isolated thread context.
    This sidesteps 'ContextVar is already entered' errors inherent to nest_asyncio
    by entirely isolating the execution context into a pristine event loop.
    """
    def _run():
        return asyncio.run(coro)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(_run).result()
