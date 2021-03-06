import typing

__all__ = ['Graph', 'Digraph']

__ATTR_DICT = typing.Optional[typing.Dict[str, str]]


class Base(object):

    @property
    def format(self) -> str:
        ...

    @format.setter
    def format(self, format: str) -> None:
        ...

    @property
    def engine(self) -> str:
        ...

    @engine.setter
    def engine(self, engine: str) -> None:
        ...

    @property
    def encoding(self) -> typing.Any:
        ...

    @encoding.setter
    def encoding(self, encoding: typing.Any):
        ...

    def copy(self) -> typing.Any:
        ...


class File(Base):
    directory: str = ...

    def __init__(self, filename: typing.Optional[str] = ..., directory: typing.Optional[str] = ...,
                 format: typing.Optional[str] = ..., engine: typing.Optional[str] = ..., encoding: typing.Any = ...):
        ...

    def pipe(self, format: typing.Optional[str] = ..., renderer: typing.Optional[str] = ...,
             formatter: typing.Optional[str] = ..., quiet: bool = ...) -> typing.Any:
        ...

    @property
    def filepath(self) -> str:
        ...

    def save(self, filename: typing.Optional[str] = ..., directory: typing.Optional[str] = ...) -> str:
        ...

    def render(self, filename: typing.Optional[str] = ..., directory: typing.Optional[str] = ...,
               view: bool = ..., cleanup: bool = ..., format: typing.Optional[str] = ...,
               renderer: typing.Optional[str] = ..., formatter: typing.Optional[str] = ...,
               quiet: bool = ..., quiet_view: bool = ...) -> str:
        ...

    def view(self, filename: typing.Optional[str] = ..., directory: typing.Optional[str] = ..., cleanup: bool = ...,
             quiet: bool = ..., quiet_view: bool = ...) -> str:
        ...


class Dot(File):
    def __init__(self, name: typing.Optional[str] = ..., comment: typing.Optional[str] = ...,
                 filename: typing.Optional[str] = ..., directory: typing.Optional[str] = ...,
                 format: typing.Optional[str] = ..., engine: typing.Optional[str] = ...,
                 encoding: typing.Any = ...,
                 graph_attr: __ATTR_DICT = ..., node_attr: __ATTR_DICT = ...,
                 edge_attr: __ATTR_DICT = ..., body: typing.Any = ...,
                 strict: bool = ...):
        ...

    def clear(self, keep_attrs: bool = ...) -> None:
        ...

    def __iter__(self, subgraph: bool = ...) -> str:
        ...

    def __str__(self) -> str:
        ...

    def node(self, name: str, label: typing.Optional[str] = ...,
             _attributes: typing.Any = ..., **attrs: typing.Optional[str]) -> None:
        ...

    def edge(self, tail_name: str, head_name: str, label: typing.Optional[str] = ...,
             _attributes: typing.Any = ..., **attrs: typing.Optional[str]) -> None:
        ...

    def edges(self, tail_head_iter: typing.Iterable[typing.Tuple[str, str]]) -> None:
        ...

    def attr(self, kw: typing.Optional[str] = ...,
             _attributes: typing.Any = ..., **attrs: typing.Optional[str]) -> None:
        ...

    def subgraph(self, graph: typing.Optional[Dot] = ...,
                 name: typing.Optional[str] = ..., comment: typing.Optional[str] = ...,
                 graph_attr: __ATTR_DICT = ..., node_attr: __ATTR_DICT = ...,
                 edge_attr: __ATTR_DICT = ..., body: typing.Any = ...) -> typing.Optional['SubgraphContext']:
        ...


class SubgraphContext:
    def __init__(self, parent: Dot, kwargs: typing.Dict[str, typing.Any]):
        ...

    def __enter__(self):
        ...

    def __exit__(self, type_: typing.Any, value: typing.Any, traceback: typing.Any):
        ...


class Graph(Dot):
    @property
    def directed(self) -> bool:
        ...


class Digraph(Dot):
    @property
    def directed(self) -> bool:
        ...
