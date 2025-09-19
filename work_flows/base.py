# work_flows/base.py
# Contains the core Workflow class that manages the event-driven system 

import asyncio 
from collections import defaultdict
from typing import Callable, Awaitable, Any
from events import StopEvent

# Tyoe alias for an event handler
EventHandler = Callable[..., Awaitable[Any]]

class Workflow:
    """
    A base class for creating event-driven workflow.
    Inspired by deep research notebook. This class manages an event loop, listeners, and the execution of async tasks.
    """

    def __init__(self, timeout: int | None = None):
        self.event_queue = asyncio.Queue()
        self.listeners = defaultdict(list)
        self.tasks = []
        self.timeout = timeout 
        self.context = {} # Shared dictionary to store state across the workflow 

    def add_listener(self, event_type: Any, handler: EventHandler):
        """ Registers an event handler for a given event type. """
        self.listeners[event_type].append(handler)

    def on_event(self, event_type: Any) -> EventHandler:
        """ A decorator to register a method as an event listener. """
        def decorator(handler: EventHandler) -> EventHandler:
            self.add_listener(event_type, handler)
            return handler 
        return decorator 
    
    async def dispatch(self, event: Any):
        """ Adds an event to the event queue to be processed. """
        await self.event_queue.put(event)

    async def process_events(self):
        """ The main event loop that processes events from the queue. """
        while True:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=self.timeout)
                if isinstance(event, StopEvent):
                    print(str(event.result))
                    print("StopEvent received. Ending Workflow.")
                    break
                    
                event_type = type(event)
                if event_type in self.listeners:
                    for handler in self.listeners[event_type]:
                        # Create a new task for each handler
                        task = asyncio.create_task(handler(event))
                        self.tasks.append(task)
            except asyncio.TimeoutError:
                print("Workflow timed out.")
                break
        if self.tasks:
            await asyncio.gather(*self.tasks)
    
    async def run(self, initial_event: Any, **kwargs):
        """ Starts the workflow with an initial event and returns when it's complete. """
        self.context = kwargs # Load initial state into context
        processor_task = asyncio.create_task(self.process_events())
        await self.dispatch(initial_event)
        await processor_task
        print("Workflow complete.")