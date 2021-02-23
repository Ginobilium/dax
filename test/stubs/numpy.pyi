import typing
import collections.abc

__all__ = ['generic', 'character', 'number', 'integer', 'int32', 'int64', 'floating', 'bool_', 'ndarray',
           'array', 'zeros', 'ones', 'empty', 'full', 'arange', 'linspace', 'logspace',
           'issubdtype', 'ndenumerate',
           'prod']


# noinspection PyPep8Naming
class generic(object):
    ...


# noinspection PyPep8Naming
class character(generic):
    ...


# noinspection PyPep8Naming
class number(generic):
    ...


# noinspection PyPep8Naming
class integer(int, number):
    ...


# noinspection PyPep8Naming
class inexact(number):
    ...


# int32, int64, and int_ have the same type properties as a Python int
int32 = int
int64 = int
int_ = int

# floating and float_ have the same type properties as a Python float
floating = float
float_ = float

# bool as the same type properties as a Python bool
bool_ = bool

# Type variable for element types
__E_T = typing.TypeVar('__E_T', bool, int, float, int32, int64, typing.Sequence[bool], typing.Sequence[int],
                       typing.Sequence[float], typing.Sequence[int32], typing.Sequence[int64])
# Type variable for argument types
__A_T = typing.TypeVar('__A_T', bool, int, float, int32, int64, typing.Sequence[bool], typing.Sequence[int],
                       typing.Sequence[float], typing.Sequence[int32], typing.Sequence[int64])
__AXIS_T = typing.Union[None, int, typing.Tuple[int, ...]]
__SHAPE_TUPLE_T = typing.Tuple[int, ...]
__SHAPE_T = typing.Union[int, __SHAPE_TUPLE_T]


# noinspection PyPep8Naming
class ndarray(collections.abc.Sequence, typing.Generic[__E_T]):

    def __init__(self, shape: __SHAPE_TUPLE_T, dtype: typing.Optional[type] = ..., buffer: typing.Any = ...,
                 offset: int = ..., strides: typing.Optional[typing.Tuple[int]] = ...,
                 order: typing.Optional[str] = ...):
        ...

    @property
    def T(self) -> ndarray[__E_T]:
        ...

    @property
    def data(self) -> memoryview:
        ...

    @property
    def dtype(self) -> type:
        ...

    @property
    def flags(self) -> typing.Dict[str, bool]:
        ...

    @property
    def flat(self) -> typing.Iterator[__E_T]:
        ...

    @property
    def imag(self) -> ndarray[__E_T]:
        ...

    @property
    def real(self) -> ndarray[__E_T]:
        ...

    @property
    def size(self) -> int:
        ...

    @property
    def itemsize(self) -> int:
        ...

    @property
    def nbytes(self) -> int:
        ...

    @property
    def ndim(self) -> int:
        ...

    @property
    def shape(self) -> typing.Tuple[int, ...]:
        ...

    @property
    def strides(self) -> typing.Tuple[int]:
        ...

    def sort(self, axis: typing.Optional[int] = ..., kind: typing.Optional[str] = ...,
             order: typing.Union[None, str, typing.List[str]] = ...):
        ...

    def argsort(self, axis: typing.Optional[int] = ..., kind: typing.Optional[str] = ...,
                order: typing.Union[None, str, typing.Sequence[str]] = ...):
        ...

    def mean(self, axis: __AXIS_T = ..., dtype: typing.Optional[type] = ...,
             out: typing.Optional['ndarray[__E_T]'] = ..., keepdims: bool = ...):
        ...

    def transpose(self, *axes: typing.Union[typing.Tuple[int, ...], int]) -> ndarray[__E_T]:
        ...

    @typing.overload
    def __getitem__(self, i: int) -> __E_T:
        ...

    @typing.overload
    def __getitem__(self, s: slice) -> ndarray[__E_T]:
        ...

    def __len__(self) -> int:
        ...

    def __add__(self, other: __A_T) -> ndarray[__A_T]:
        ...

    def __iadd__(self, other: __A_T) -> ndarray[__E_T]:
        ...

    def __sub__(self, other: __A_T) -> ndarray[__A_T]:
        ...

    def __isub__(self, other: __A_T) -> ndarray[__E_T]:
        ...

    def __mul__(self, other: __A_T) -> ndarray[__A_T]:
        ...

    def __imul__(self, other: __A_T) -> ndarray[__E_T]:
        ...

    def __truediv__(self, other: __A_T) -> ndarray[__A_T]:
        ...

    def __itruediv__(self, other: __A_T) -> ndarray[__E_T]:
        ...


def array(object: typing.Sequence[__E_T], dtype: typing.Optional[type] = ..., copy: bool = ...,
          order: str = ..., subok: bool = ..., ndmin: int = ...) -> ndarray[__E_T]:
    ...


def zeros(shape: __SHAPE_T, dtype: typing.Optional[typing.Type[__E_T]] = ..., order: str = ...) -> ndarray[__E_T]:
    ...


def ones(shape: __SHAPE_T, dtype: typing.Optional[typing.Type[__E_T]] = ..., order: str = ...) -> ndarray[__E_T]:
    ...


def empty(shape: __SHAPE_T, dtype: typing.Optional[typing.Type[__E_T]] = ..., order: str = ...) -> ndarray[__E_T]:
    ...


def full(shape: __SHAPE_T, fill_value: __E_T, dtype: typing.Optional[typing.Type[__E_T]] = ...,
         order: str = ...) -> ndarray[__E_T]:
    ...


@typing.overload
def arange(stop: __E_T, dtype: typing.Optional[type] = ...) -> ndarray[__E_T]:
    ...


@typing.overload
def arange(start: __E_T, stop: __E_T, step: typing.Optional[__E_T] = ...,
           dtype: typing.Optional[type] = ...) -> ndarray[__E_T]:
    ...


def linspace(start: __E_T, stop: __E_T, num: int = ..., endpoint: bool = ..., retstep: bool = ...,
             dtype: typing.Optional[type] = ..., axis: int = ...) -> ndarray[__E_T]:
    ...


def logspace(start: __E_T, stop: __E_T, num: int = ..., endpoint: bool = ..., base: __E_T = ...,
             dtype: typing.Optional[type] = ..., axis: int = ...) -> ndarray[__E_T]:
    ...


def asarray(a: typing.Sequence[__E_T], dtype: typing.Optional[type] = ...,
            order: typing.Optional[str] = ...) -> ndarray[__E_T]:
    ...


def column_stack(tup: typing.Sequence[ndarray[__E_T]]) -> ndarray[__E_T]:
    ...


def concatenate(*arrays: typing.Sequence[__E_T], axis: typing.Optional[int] = ...,
                out: typing.Optional[ndarray[__E_T]] = ...) -> ndarray[__E_T]:
    ...


def issubdtype(arg1: typing.Union[type, str], arg2: typing.Union[type, str]) -> bool:
    ...


def ndenumerate(arr: ndarray[__E_T]) -> typing.Iterator[typing.Tuple[typing.Tuple[int, ...], __E_T]]:
    ...


def prod(a: typing.Sequence[__E_T], axis: __AXIS_T = ..., dtype: typing.Optional[type] = ...,
         out: typing.Optional[ndarray[__E_T]] = ..., keepdims: typing.Optional[bool] = ...,
         initial: typing.Optional[__E_T] = ..., where: typing.Optional[typing.Sequence[bool]] = ...) -> __E_T:
    ...
