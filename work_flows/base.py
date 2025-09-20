# work_flows/base.py
# Contains the core Workflow class that manages the event-driven system 

import asyncio 
from collections import defaultdict
from typing import Callable, Awaitable, Any
from events import StopEvent

# Type alias for an event handler
EventHandler = Callable[..., Awaitable[Any]]

class Workflow:
    """
    A base class for creating event-driven workflows.
    Inspired by the LlamaIndex deep research notebook. This class manages an
    event loop, listeners, and the execution of async tasks.
    """

    def __init__(self, timeout: int | None = None, **kwargs):
        self.event_queue = asyncio.Queue()
        self.listeners = defaultdict(list)
        self.tasks = []
        self.timeout = timeout 
        self.context = kwargs # Use kwargs to initialize context

    def add_listener(self, event_type: Any, handler: EventHandler):
        """ Registers an event handler for a given event type. """
        self.listeners[event_type].append(handler)
    
    async def dispatch(self, event: Any):
        """ Adds an event to the event queue to be processed. """
        await self.event_queue.put(event)

    async def process_events(self):
        """ The main event loop that processes events from the queue. """
        while True:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=self.timeout)
                if isinstance(event, StopEvent):
                    print("StopEvent received. Ending Workflow.")
                    return event.result
                    
                event_type = type(event)
                if event_type in self.listeners:
                    for handler in self.listeners[event_type]:
                        task = asyncio.create_task(handler(event))
                        self.tasks.append(task)
            except asyncio.TimeoutError:
                print("Workflow timed out.")
                break
        
        if self.tasks:
            await asyncio.gather(*self.tasks)
        
        return None # Return None if workflow times out

    async def run(self, initial_event: Any, **kwargs):
        """ Starts the workflow with an initial event and returns when it's complete. """
        if kwargs:
            self.context.update(kwargs) # Update context with any new run-specific data
        
        processor_task = asyncio.create_task(self.process_events())
        await self.dispatch(initial_event)
        
        # --- Capture and return the result from the processor task ---
        final_result = await processor_task
        print("Workflow complete.")
        return final_result