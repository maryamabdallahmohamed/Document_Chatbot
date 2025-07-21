import logging
class TaskProcessor:
    """Context class that uses different task strategies."""
    
    def __init__(self, strategy=None):  
        self._strategy = strategy
        self.logger = logging.getLogger(self.__class__.__name__)
        if strategy:
            self.logger.info(f"🎯 TaskProcessor initialized with strategy: {strategy.__class__.__name__}")
        else:
            self.logger.info("🎯 TaskProcessor initialized without strategy")
    
    @property
    def strategy(self):
        return self._strategy
         
    @strategy.setter
    def strategy(self, strategy):
        old_strategy = self._strategy.__class__.__name__ if self._strategy else "None"
        new_strategy = strategy.__class__.__name__ if strategy else "None"
        self._strategy = strategy
        self.logger.info(f"🔄 Strategy changed: {old_strategy} → {new_strategy}")
         
    def execute_task(self, *args, **kwargs):
        if self._strategy is None:     
            self.logger.error("❌ No strategy set for task execution")
            raise ValueError("No strategy set")
        
        strategy_name = self._strategy.__class__.__name__
        self.logger.info(f"🚀 Executing task with {strategy_name}")
        
        try:
            result = self._strategy.run(*args, **kwargs)
            self.logger.info(f"✅ Task completed successfully with {strategy_name}")
            return result
        except Exception as e:
            self.logger.error(f"❌ Task failed with {strategy_name}: {str(e)}")
            raise
         
