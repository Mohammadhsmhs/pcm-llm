"""
Application bootstrap module for PCM-LLM.
Handles dependency injection setup and application initialization.
"""

from typing import Optional
from core.container import container, ServiceLocator, IContainer
from core.config import IConfigProvider, CentralizedConfigProvider, settings
from core.llm_factory import ILLMFactory, LLMFactory
from core.benchmark_service import IBenchmarkService, BenchmarkService
from utils import RunInfoLogger


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
        
        # LLM factory - use a factory function to ensure dependency is resolved
        def create_llm_factory(c: IContainer) -> ILLMFactory:
            return LLMFactory(config_provider=c.resolve(IConfigProvider))
        
        container.register_singleton(ILLMFactory, create_llm_factory)
        
    def _register_business_services(self) -> None:
        """Register business logic services."""
        # Benchmark service
        def create_benchmark_service(c: IContainer) -> IBenchmarkService:
            return BenchmarkService(
                run_info_logger=c.resolve(RunInfoLogger),
                llm_factory=c.resolve(ILLMFactory),
            )
        container.register(IBenchmarkService, create_benchmark_service)
    
    def _register_utilities(self) -> None:
        """Register utility services."""
        # Run info logger - create with proper settings
        def create_run_info_logger():
            return RunInfoLogger(log_dir=settings.paths.logs_dir)
        
        container.register_singleton(RunInfoLogger, lambda: create_run_info_logger())
    
    def get_config_provider(self) -> IConfigProvider:
        """Get the configuration provider."""
        return ServiceLocator.resolve(IConfigProvider)
    
    def get_llm_factory(self) -> ILLMFactory:
        """Get the LLM factory."""
        return ServiceLocator.resolve(ILLMFactory)
    
    def get_benchmark_service(self) -> IBenchmarkService:
        """Get the benchmark service."""
        return ServiceLocator.resolve(IBenchmarkService)
    
    def get_run_info_logger(self, **kwargs) -> RunInfoLogger:
        """Get a run info logger instance."""
        return ServiceLocator.resolve(RunInfoLogger)


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
