"""
Dependency injection container for PCM-LLM.
Follows the Dependency Inversion Principle and provides clean dependency management.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type


class IContainer(ABC):
    """Interface for dependency injection container."""

    @abstractmethod
    def register(self, interface: Type, implementation: Type) -> None:
        """Register an implementation for an interface."""
        pass

    @abstractmethod
    def register_singleton(self, interface: Type, implementation: Type) -> None:
        """Register a singleton implementation for an interface."""
        pass

    @abstractmethod
    def resolve(self, interface: Type) -> Any:
        """Resolve an implementation for an interface."""
        pass

    @abstractmethod
    def get_instance(self, interface: Type) -> Any:
        """Get an instance of an interface (creates new instance each time)."""
        pass


class Container(IContainer):
    """Simple dependency injection container implementation."""

    def __init__(self):
        self._registrations: Dict[Type, Type] = {}
        self._singletons: Dict[Type, Any] = {}
        self._instances: Dict[Type, Any] = {}

    def register(self, interface: Type, implementation: Type) -> None:
        """Register an implementation for an interface."""
        self._registrations[interface] = implementation

    def register_singleton(self, interface: Type, implementation: Type) -> None:
        """Register a singleton implementation for an interface."""
        self._registrations[interface] = implementation
        # Create the singleton instance immediately
        self._singletons[interface] = self._create_instance(implementation)

    def resolve(self, interface: Type) -> Any:
        """Resolve an implementation for an interface."""
        if interface in self._singletons:
            return self._singletons[interface]

        if interface in self._registrations:
            return self._create_instance(self._registrations[interface])

        raise ValueError(f"No implementation registered for interface: {interface}")

    def get_instance(self, interface: Type) -> Any:
        """Get an instance of an interface (creates new instance each time)."""
        if interface in self._registrations:
            return self._create_instance(self._registrations[interface])

        raise ValueError(f"No implementation registered for interface: {interface}")

    def _create_instance(self, implementation: Type) -> Any:
        """Create an instance of the implementation class."""
        try:
            # Try to create instance without parameters first
            return implementation()
        except TypeError:
            # If that fails, try to create with container as parameter
            try:
                return implementation(self)
            except TypeError:
                # If that also fails, try to create with dependencies
                try:
                    return self._create_with_dependencies(implementation)
                except Exception:
                    # Last resort: try to create with no parameters
                    return implementation()

    def _create_with_dependencies(self, implementation: Type) -> Any:
        """Create instance by injecting dependencies from container."""
        # Get the constructor signature
        import inspect

        sig = inspect.signature(implementation.__init__)

        # Build arguments dictionary
        args = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            # Try to resolve the parameter type from the container
            param_type = param.annotation
            if param_type != inspect.Parameter.empty:
                try:
                    args[param_name] = self.resolve(param_type)
                except ValueError:
                    # If we can't resolve the type, try to get an instance
                    try:
                        args[param_name] = self.get_instance(param_type)
                    except ValueError:
                        # Skip this parameter if we can't resolve it
                        pass

        # Create instance with resolved dependencies
        return implementation(**args)


class ServiceLocator:
    """Service locator pattern for accessing global services."""

    _container: Optional[IContainer] = None

    @classmethod
    def set_container(cls, container: IContainer) -> None:
        """Set the global container instance."""
        cls._container = container

    @classmethod
    def get_container(cls) -> IContainer:
        """Get the global container instance."""
        if cls._container is None:
            raise RuntimeError("Container not initialized. Call set_container() first.")
        return cls._container

    @classmethod
    def resolve(cls, interface: Type) -> Any:
        """Resolve a dependency using the global container."""
        return cls.get_container().resolve(interface)

    @classmethod
    def get_instance(cls, interface: Type) -> Any:
        """Get an instance using the global container."""
        return cls.get_container().get_instance(interface)


# Global container instance
container = Container()
ServiceLocator.set_container(container)
