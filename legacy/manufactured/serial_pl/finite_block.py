"""S1 five-node fixture extracted from finite_block.py.

Function/class bodies are unchanged source extracts.
"""
from __future__ import annotations

from dataclasses import dataclass

from typing import Callable

import copy, math, time

from types import MappingProxyType

from collections.abc import Mapping

import numpy as np

from scipy.linalg import cho_factor, cho_solve, eigh

from scipy.special import erfcx

@dataclass
class Model:
    kind: str='nonlinear'
    delta: float=.02
    theta: tuple=(1., .1, .01)
    beta: float=.6
    def __post_init__(self):
        self.delta=real_scalar(self.delta,'delta',0.,1.,True)
        self.beta=real_scalar(self.beta,'beta',0.)
        self.n=5; self.r=3; self.s=2; self.T=1.; self.alpha=.5
        self.theta=np.array(numeric_array(self.theta,'theta',(3,)),dtype=float,copy=True)
        if self.theta.shape!=(3,) or np.any(self.theta<=0) or self.theta.max()!=1 or not 0<self.delta<=1:
            raise ValueError('invalid masses')
        perm=np.array([1,2,3,0,4]); edges=[(i,i+1) for i in range(4)]
        if self.kind=='linear':
            self.beta=0.; w=np.array([2.5,4.,4.5,4.,2.5]); losses=w.copy()
            g=np.array([w[i]*w[j] for i,j in edges]); self.p1=np.zeros(5)
        elif self.kind=='nonlinear':
            w=np.ones(5); losses=np.array([1.5,.5,.5,.5,1.5]);
            edges += [(1,3),(2,4)]; g=np.r_[np.ones(4),.7,.5]; self.p1=.2*np.eye(5)[3]
        else: raise ValueError('invalid model kind')
        inc=np.zeros((5,len(edges)))
        for j,(a,b) in enumerate(edges): inc[a,j]=1; inc[b,j]=-1
        self.inc=inc[perm]; self.g=g; self.losses=losses[perm]
        self.A0=np.diag(self.losses)+(self.inc*g)@self.inc.T
        self.p0=(w*np.array([1.,0,0,0,0]))[perm]
        self.scale=w[perm]; self.u0=np.array([.1,.8,-.5,.6,.2])[perm]/self.scale
        self.mass=np.r_[self.delta*self.theta,1.,1.]
        validate_model(self)
    def F(self,u):
        u=np.asarray(u); z=u@self.inc
        return u@self.A0.T+(self.beta*self.g*np.tanh(z))@self.inc.T
    def J(self,u):
        z=np.asarray(u)@self.inc
        return self.A0+(self.inc*(self.beta*self.g*(1-np.tanh(z)**2)))@self.inc.T
    def dJ(self,u,v):
        z=np.asarray(u)@self.inc; dz=np.asarray(v)@self.inc
        dg=-2*self.beta*self.g*np.tanh(z)*(1-np.tanh(z)**2)*dz
        return (self.inc*dg)@self.inc.T
    def p(self,t): return self.p0+float(t)*self.p1

_MODEL_FIELDS = frozenset(('kind','delta','theta','beta','n','r','s','T','alpha',
                          'inc','g','losses','A0','p0','p1','scale','u0','mass'))

def real_scalar(value, name, lower=None, upper=None, strict_lower=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name}: finite real scalar required')
    if isinstance(value,np.floating) and value.dtype.itemsize>8:
        raise ValueError(f'{name}: only binary64-or-narrower scalars are supported')
    try: x = float(value)
    except (OverflowError,ValueError) as exc: raise ValueError(f'{name}: not representable as binary64') from exc
    if not math.isfinite(x):
        raise ValueError(f'{name}: finite real scalar required')
    if lower is not None and (x <= lower if strict_lower else x < lower):
        raise ValueError(f'{name}: below admissible range')
    if upper is not None and x > upper:
        raise ValueError(f'{name}: above admissible range')
    return x

def numeric_array(value, name, shape=None):
    a = np.asarray(value)
    if a.dtype.kind not in 'fiu' or (shape is not None and a.shape != shape) or not np.all(np.isfinite(a)):
        raise ValueError(f'{name}: finite real array of shape {shape} required')
    return a

def validate_model(m):
    """Reject unsupported or inconsistent data BEFORE numerical preparation.

    Only the concrete five-node Model data schema is implemented. Arbitrary
    subclasses, instance method overrides and extra attributes are unsupported.
    Changes to supported numeric data require a new cache, not an accuracy claim.
    """
    if type(m) is not Model or set(vars(m)) != _MODEL_FIELDS:
        raise ValueError('unsupported Model schema, subclass or overridden method')
    if (m.n, m.r, m.s) != (5, 3, 2):
        raise ValueError('only n=5, r=3, s=2 is implemented')
    if real_scalar(m.alpha, 'alpha') != .5 or real_scalar(m.T, 'T') != 1.:
        raise ValueError('only alpha=1/2 and T=1 are implemented')
    if m.kind not in ('linear','nonlinear'):
        raise ValueError('unsupported kind')
    d = real_scalar(m.delta, 'delta', 0., 1., True)
    beta = real_scalar(m.beta, 'beta', 0.)
    if m.kind == 'linear' and beta != 0.:
        raise ValueError('linear kind requires beta=0')
    # The finite backend must be able to form its scaled memory argument.
    if d*d == 0. or not math.isfinite(1./(d*d)):
        raise ValueError('delta**2 is outside the floating-point backend range')
    e = 4 if m.kind == 'linear' else 6
    for name, shape in {'theta':(3,), 'inc':(5,e), 'g':(e,), 'losses':(5,),
                        'A0':(5,5), 'p0':(5,), 'p1':(5,), 'scale':(5,),
                        'u0':(5,), 'mass':(5,)}.items():
        if not isinstance(getattr(m,name),np.ndarray):
            raise ValueError(f'{name}: ndarray required')
        numeric_array(getattr(m,name),name,shape)
        if getattr(m,name).dtype!=np.dtype('float64'):
            raise ValueError(f'{name}: Model storage must use float64')
    if np.any(m.theta <= 0.) or np.max(m.theta) != 1.:
        raise ValueError('theta must be positive and normalized by max(theta)=1')
    if np.any(m.mass <= 0.) or not np.array_equal(m.mass[:3],d*m.theta):
        raise ValueError('positive masses and exact mass_f=delta*theta required')
    if not np.array_equal(m.mass[3:],np.ones(2)):
        raise ValueError('only retained mass M_s=I is implemented')
    if np.any(m.scale <= 0.) or np.any(m.losses <= 0.) or np.any(m.g <= 0.):
        raise ValueError('positive scale, losses and conductances required')
    # Keep the existing two graph topologies; no graph extension is claimed.
    edges=[(i,i+1) for i in range(4)]
    if m.kind == 'nonlinear': edges += [(1,3),(2,4)]
    incidence=np.zeros((5,e))
    for j,(a,b) in enumerate(edges): incidence[a,j]=1.;incidence[b,j]=-1.
    if not np.array_equal(m.inc,incidence[np.array([1,2,3,0,4])]):
        raise ValueError('unsupported graph incidence')
    with np.errstate(over='raise', invalid='raise'):
        try: expected=np.diag(m.losses)+(m.inc*m.g)@m.inc.T
        except FloatingPointError as exc: raise ValueError('graph matrix overflows') from exc
    if not np.array_equal(m.A0,expected):
        raise ValueError('A0 inconsistent with incidence, losses and conductances')
    return m

from typing import NamedTuple
