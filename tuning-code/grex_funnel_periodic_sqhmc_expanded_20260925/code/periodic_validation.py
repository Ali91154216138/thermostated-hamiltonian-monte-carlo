"""Check the periodic THC integrator on Neal's funnel."""
import itertools
import math
import numpy as np

def potential(q,parameters):
    return q[0]**2/(2*parameters[0])+parameters[1]*q[0]/(2*parameters[2])+.5*np.exp(-q[0])*(q[1:]@q[1:])

def gradient(q,parameters):
    v=np.exp(-q[0]); g=np.empty_like(q)
    g[0]=q[0]/parameters[0]+parameters[1]/(2*parameters[2])-.5*v*(q[1:]@q[1:])
    g[1:]=v*q[1:]
    return g

def reference(q,p,s,ps,beta,alpha,phi,a,Q,sigma,eps,L,parameters,inverse):
    """Unfused, vectorized subflows; independent from native cached-loop code."""
    q=q.copy(); p=p.copy(); w=1/(sigma*math.sqrt(beta*Q))
    angle=eps*w/2; cosine=math.cos(angle); sine=math.sin(angle)
    def rotate(s,ps):
        return cosine*s+sine*ps/(Q*w),cosine*ps-sine*Q*w*s
    def coupling(s):
        h=1+alpha*math.cos(s/a-phi)
        dh=-alpha/a*math.sin(s/a-phi)
        return h,dh/h
    for _ in range(L):
        h,dlog=coupling(s)
        p-=eps/2*gradient(q,parameters); ps-=eps/2*len(q)/beta*dlog
        s,ps=rotate(s,ps)
        h,dlog=coupling(s); velocity=inverse@p/h**2
        q+=eps*velocity; ps+=eps*dlog*(p@velocity)
        s,ps=rotate(s,ps)
        h,dlog=coupling(s)
        p-=eps/2*gradient(q,parameters); ps-=eps/2*len(q)/beta*dlog
    return q,p,s,ps

def energy(q,p,s,ps,beta,alpha,phi,a,Q,sigma,parameters,inverse):
    h=1+alpha*math.cos(s/a-phi)
    return potential(q,parameters)+.5*(p@inverse@p)/h**2+len(q)/beta*math.log(h)+ps**2/(2*Q)+s**2/(2*beta*sigma**2)

def validate_periodic(mod,prepared):
    if not mod.NUMBA_AVAILABLE: raise RuntimeError('Numba is required for production')
    common=prepared.common; params=common.target_parameters
    q=np.array([.17,-.11,.08,.04,-.09,.13]); p=np.array([.31,-.24,.07,.12,-.05,.09])
    s,ps=.23,-.16
    M=np.eye(6); M[0,0]=1.4; M[0,1]=M[1,0]=.12; inverse=np.linalg.inv(M)
    choices=list(itertools.product((.025,.95),(0.,math.pi/2),(.0625,16.),(.003125,128.)))
    choices.append((.3,.37,.75,.5))
    for alpha,phi,a,Q in choices:
        beta=.7; eps=.00001; L=7; sigma=1.3
        g=gradient(q,params)
        got=mod._thc_proposal(q,p,s,ps,g,beta,alpha,phi,a,Q,sigma,eps,L,params,inverse,False)
        expected=reference(q,p,s,ps,beta,alpha,phi,a,Q,sigma,eps,L,params,inverse)
        assert got[6] and got[5]==L, ('native proposal/gradient count',alpha,phi,a,Q)
        for j in range(4):
            assert np.allclose(got[j],expected[j],rtol=2e-11,atol=2e-12),('independent split',j,alpha,phi,a,Q)
        native_H=mod._thc_hamiltonian(q,p,s,ps,beta,alpha,phi,a,Q,sigma,params,inverse,False)
        assert math.isclose(native_H,energy(q,p,s,ps,beta,alpha,phi,a,Q,sigma,params,inverse),rel_tol=2e-13,abs_tol=2e-13)
        logh,dlog=mod._log_h_and_derivative(s,alpha,phi,a)
        step=1e-6*a
        fd=(math.log(1+alpha*math.cos((s+step)/a-phi))-math.log(1+alpha*math.cos((s-step)/a-phi)))/(2*step)
        assert math.isclose(dlog,fd,rel_tol=2e-7,abs_tol=2e-8)
        assert math.isclose(math.exp(logh),1+alpha*math.cos(s/a-phi),rel_tol=2e-14)
        # Gaussian momentum integral cancels h^{-d}; no physical tempering.
        cancellation=-len(q)*logh+.5*np.linalg.slogdet(math.exp(2*logh)*M/beta)[1]
        assert math.isclose(cancellation,.5*np.linalg.slogdet(M/beta)[1],abs_tol=1e-12)
    alpha,phi,a,Q=.3,.37,.75,.5; beta=1.; sigma=1.; eps=.003; L=3
    def pack(q,p,s,ps): return np.concatenate((q,p,[s,ps]))
    state=pack(q,p,s,ps)
    def forward(z,eps=eps,L=L):
        return mod._thc_proposal(z[:6],z[6:12],z[12],z[13],gradient(z[:6],params),
            beta,alpha,phi,a,Q,sigma,eps,L,params,inverse,False)
    z1=forward(state)
    z2=forward(pack(z1[0],-z1[1],z1[2],-z1[3]))
    assert np.allclose(pack(z2[0],-z2[1],z2[2],-z2[3]),state,rtol=2e-12,atol=2e-12)
    jac=np.empty((14,14)); d=1e-6
    for j in range(14):
        plus=state.copy(); minus=state.copy(); plus[j]+=d; minus[j]-=d
        up=forward(plus); dn=forward(minus)
        jac[:,j]=(pack(*up[:4])-pack(*dn[:4]))/(2*d)
    assert abs(np.linalg.det(jac)-1)<1e-6, 'volume preservation'
    endpoints=[]; errors=[]
    initial=energy(q,p,s,ps,beta,alpha,phi,a,Q,sigma,params,inverse)
    for step,count in ((.04,10),(.02,20),(.01,40),(.0003125,1280)):
        end=forward(state,step,count); endpoints.append(pack(*end[:4]))
        errors.append(abs(energy(*end[:4],beta,alpha,phi,a,Q,sigma,params,inverse)-initial))
    phase=[np.linalg.norm(e-endpoints[-1]) for e in endpoints[:3]]
    assert 3.5<phase[0]/phase[1]<4.5 and 3.5<phase[1]/phase[2]<4.5, ('phase order',phase)
    assert 3<errors[0]/errors[1]<5 and 3<errors[1]/errors[2]<5, ('energy order',errors)
    # alpha=0 gives the already-tested constant-mass HMC path, including s rotation.
    native=mod._thc_proposal(q,p,s,ps,gradient(q,params),1.,0.,.37,.75,.5,1.,.02,20,params,inverse,False)
    bounded_zero=mod.base._thc_proposal(q,p,s,ps,gradient(q,params),1.,0.,.75,.5,1.,.02,20,params,inverse,False)
    for j in range(5): assert np.allclose(native[j],bounded_zero[j],rtol=2e-13,atol=2e-13)
    # One accepted step uses the correct physical covariance and ps variance.
    Z=np.array([[.1,.2,-.1,.05,.3,-.12]]); Zs=np.array([.24]); uniforms=np.array([0.])
    beta=.7; Q=.5; alpha=.3; phi=.37; a=.75; sigma=1.3; eps=.004; L=3
    chol=np.linalg.cholesky(M)
    h=1+alpha*math.cos(s/a-phi); p0=h*(chol@Z[0])/math.sqrt(beta); ps0=math.sqrt(Q/beta)*Zs[0]
    expected=reference(q,p0,s,ps0,beta,alpha,phi,a,Q,sigma,eps,L,params,inverse)
    out=mod._simulate_thc_chain(Z,Zs,uniforms,q,s,beta,alpha,phi,a,Q,sigma,eps,L,0,1,params,chol,inverse,False)
    assert out[2]==1 and np.allclose(out[0][0],expected[0],rtol=2e-12,atol=2e-12)
    assert math.isclose(out[1][0],expected[2],rel_tol=2e-12,abs_tol=2e-12)
    assert out[6]==L+1
    # Bad proposals must restore BOTH positions, never fabricate a new draw.
    bad=mod._simulate_thc_chain(np.full((3,6),np.nan),np.ones(3),np.full(3,.5),q,s,
        1.,.3,.37,.75,.5,1.,.02,3,1,2,params,np.eye(6),np.eye(6),True)
    assert np.array_equal(bad[0],np.tile(q,(2,1))) and np.array_equal(bad[1],np.full(2,s))
    assert bad[3]==1 and bad[4]==2
    streams=mod._make_random_streams(18092026,144,6)
    smoke=mod._simulate_thc_chain(*streams,q,s,1.,.1,0.,.75,.1,1.,.02,5,16,128,params,np.eye(6),np.eye(6),True)
    assert np.isfinite(smoke[0]).all() and np.isfinite(smoke[1]).all()
    assert smoke[5]==1+16*5 and smoke[6]==128*5 and smoke[3]==smoke[4]==0
    assert mod._simulate_thc_chain.nopython_signatures and mod._thc_proposal.nopython_signatures
    return (
        'PASS periodic: independent split and Hamiltonian at all 16 extreme corners plus interior; log-h derivative; exact marginal cancellation.',
        'PASS periodic: reversibility, numerical unit Jacobian, second-order path/energy, alpha=0 constant-mass HMC equivalence.',
        'PASS periodic: mass/beta momentum refresh, full q/s rejection restoration, actual gradient counts, native finite smoke chain.',
    )
