"""
Application bootstrap module for PCM-LLM.
Handles dependency injection setup and application initialization.
"""

from typing import Optional
from core.container import container, ServiceLocator
from core.config import IConfigProvider, CentralizedConfigProvider
from core.llm_factory import ILLMFactory, LLMFactory
from core.benchmark_service import IBenchmarkService, BenchmarkService, DataLoaderAdapter, IDataLoader, ILogger, IRunInfoLogger
from utils.logger import BenchmarkLogger
from utils.run_info_logger import RunInfoLogger


class ApplicationBootstrap:
    """Bootstrap class for initializing the PCM-LLM application."""
    
    def __init__(self):
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the application with all dependencies."""
        if self._initialized:
            return
        
        # Register core services
        self._register_core_services()
        
        # Register business services
        self._register_business_services()
        
        # Register utilities
        self._register_utilities()
        
        self._initialized = True
    
    def _register_core_services(self) -> None:
        """Register core application services."""
        # Configuration provider
        config_provider = CentralizedConfigProvider()
        container.register_singleton(IConfigProvider, lambda: config_provider)
        
        # LLM factory
        llm_factory = LLMFactory(config_provider)
        container.register_singleton(ILLMFactory, lambda: llm_factory)
        
        # Data loader
        data_loader = DataLoaderAdapter()
        container.register_singleton(IDataLoader, lambda: data_loader)
    
    def _register_business_services(self) -> None:
        """Register business logic services."""
        # Benchmark service
        container.register(IBenchmarkService, BenchmarkService)
    
    def _register_utilities(self) -> None:
        """Register utility services."""
        # Logger
        container.register(ILogger, BenchmarkLogger)
        
        # Run info logger
        container.register(IRunInfoLogger, RunInfoLogger)
    
    def get_config_provider(self) -> IConfigProvider:
        """Get the configuration provider."""
        return ServiceLocator.resolve(IConfigProvider)
    
    def get_llm_factory(self) -> ILLMFactory:
        """Get the LLM factory."""
        return ServiceLocator.resolve(ILLMFactory)
    
    def get_data_loader(self) -> IDataLoader:
        """Get the data loader."""
        return ServiceLocator.resolve(IDataLoader)
    
    def get_benchmark_service(self) -> IBenchmarkService:
        """Get the benchmark service."""
        return ServiceLocator.get_instance(IBenchmarkService)
    
    def get_logger(self, **kwargs) -> ILogger:
        """Get a logger instance."""
        return ServiceLocator.get_instance(ILogger)
    
    def get_run_info_logger(self, **kwargs) -> IRunInfoLogger:
        """Get a run info logger instance."""
        return ServiceLocator.get_instance(IRunInfoLogger)


# Global bootstrap instance
bootstrap = ApplicationBootstrap()


def initialize_app() -> ApplicationBootstrap:
    """Initialize the application and return the bootstrap instance."""
    bootstrap.initialize()
    return bootstrap


def get_app() -> ApplicationBootstrap:
    """Get the application bootstrap instance."""
    if not bootstrap._initialized:
        bootstrap.initialize()
    return bootstrap
