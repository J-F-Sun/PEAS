"""Physical domain sampling; no PEAS feature maps or backward formulation."""
import numpy as np
from scipy.special import ndtr
from scipy.stats import ncx2


def contains(x, c):
    x = np.atleast_2d(x)
    g = c['geometry']
    inside = np.linalg.norm(x - g['rx_center'], axis=1) > g['rx_radius']
    if c['scene'] in ('concentric', 'eccentric'):
        inside &= np.linalg.norm(x-g['outer_center'], axis=1) < g['outer_radius']
    elif c['scene'] == 'pipe':
        inside &= (np.linalg.norm(x[:, :2], axis=1) < g['pipe_radius']) & (x[:, 2] > g['z_min']) & (x[:, 2] < g['z_max'])
    else:
        inside &= np.linalg.norm(x-g['rx_center'], axis=1) < c['numerics']['outer_cutoff']
    return inside


def sigma_for(source, c):
    g = c['geometry']
    gap = np.linalg.norm(source-np.array(g['rx_center']))-g['rx_radius']
    if c['scene'] in ('concentric', 'eccentric'):
        gap = min(gap, g['outer_radius']-np.linalg.norm(source-np.array(g['outer_center'])))
    if c['scene'] == 'pipe':
        gap = min(gap, g['pipe_radius']-np.linalg.norm(source[:2]), source[2]-g['z_min'], g['z_max']-source[2])
    if gap <= 0:
        raise ValueError('Source outside physical fluid domain')
    return min(c['source']['sigma_max'], gap/c['source']['boundary_sigma_margin'])


def normalization(source, sigma, c):
    g = c['geometry']
    inner = ncx2.cdf((g['rx_radius']/sigma)**2, 3, np.sum((source-g['rx_center'])**2)/sigma**2)
    if c['scene'] == 'pipe':
        outer = ncx2.cdf((g['pipe_radius']/sigma)**2, 2, np.sum(source[:2]**2)/sigma**2)
        outer *= ndtr((g['z_max']-source[2])/sigma)-ndtr((g['z_min']-source[2])/sigma)
    else:
        center = g.get('outer_center', g['rx_center'])
        radius = g.get('outer_radius', c['numerics']['outer_cutoff'])
        outer = ncx2.cdf((radius/sigma)**2, 3, np.sum((source-center)**2)/sigma**2)
    return float(outer-inner)


def initial_density(x, source, sigma, c):
    z = normalization(source, sigma, c)
    if not z > 1e-8:
        raise ValueError('Gaussian normalization vanishes')
    if c['representation'] == 'radial':
        # Exact spherical average of a displaced 3-D isotropic Gaussian.
        r = np.asarray(x).reshape(-1)
        r0 = np.linalg.norm(source-np.array(c['geometry']['rx_center']))
        v = (np.exp(-.5*((r-r0)/sigma)**2)-np.exp(-.5*((r+r0)/sigma)**2))
        return v/((2*np.pi*sigma*sigma)**1.5 * (2*r*r0/sigma**2) * z)
    return np.exp(-np.sum((x-source)**2, axis=1)/(2*sigma*sigma))/((2*np.pi*sigma*sigma)**1.5*z)


def sphere_surface(center, radius, n):
    # Deterministic Fibonacci quadrature. Normals point away from sphere center.
    i = np.arange(n)
    z = 1-2*(i+.5)/n
    theta = i*np.pi*(3-np.sqrt(5))
    u = np.column_stack((np.sqrt(1-z*z)*np.cos(theta), np.sqrt(1-z*z)*np.sin(theta), z))
    return np.array(center)+radius*u, u, np.full(n, 4*np.pi*radius*radius/n)


def sample_uniform(rng, n, c):
    g = c['geometry']
    result = []
    count = 0
    while count < n:
        k = max(128, 2*(n-count))
        if c['scene'] == 'pipe':
            rho = g['pipe_radius']*np.sqrt(rng.random(k)); angle=rng.uniform(0, 2*np.pi, k)
            x = np.column_stack((rho*np.cos(angle), rho*np.sin(angle), rng.uniform(g['z_min'],g['z_max'],k)))
        else:
            radius = g.get('outer_radius', c['numerics']['outer_cutoff'])
            u=rng.normal(size=(k,3));u/=np.linalg.norm(u,axis=1)[:,None]
            x=np.array(g.get('outer_center', g['rx_center']))+radius*rng.random(k)[:,None]**(1/3)*u
        accepted=x[contains(x,c)];result.append(accepted);count+=len(accepted)
    return np.concatenate(result)[:n]


def sample_domain(c, source, sigma, rng):
    from swimpde.Domain import Domain
    q = c['sampling'];g=c['geometry'];n=q['interior_points']
    local_n=int(n*q['source_fraction'])
    if c['representation']=='radial':
        a=g['rx_radius']; R=g.get('outer_radius',c['numerics']['outer_cutoff'])
        r0=np.linalg.norm(source-g['rx_center'])
        global_r=rng.uniform(a,R,n-local_n)
        local=[];count=0
        while count<local_n:
            r=rng.normal(r0,sigma*q['source_spread'],max(64,2*(local_n-count)))
            r=r[(r>a)&(r<R)];local.append(r);count+=len(r)
        x=np.concatenate((global_r,np.concatenate(local)[:local_n] if local else []))[:,None]
        boundary=np.array([[a],[R]])
        normals=np.array([[-1.],[1.]])
        labels=np.array(['rx','outer'])
    else:
        uniform=sample_uniform(rng,n-local_n,c);local=[];count=0
        while count<local_n:
            x=rng.normal(source,sigma*q['source_spread'],(max(64,2*(local_n-count)),3))
            x=x[contains(x,c)];local.append(x);count+=len(x)
        x=np.vstack((uniform,np.concatenate(local)[:local_n] if local else np.empty((0,3))))
        k=q['surface_points'];rx,rxn,_=sphere_surface(g['rx_center'],g['rx_radius'],k)
        if c['scene'] in ('concentric', 'eccentric'):
            outer,on,_=sphere_surface(g['outer_center'],g['outer_radius'],k)
            boundary=np.vstack((rx,outer));normals=np.vstack((-rxn,on));labels=np.array(['rx']*k+['outer']*k)
        else:
            theta=rng.uniform(0,2*np.pi,k);z=rng.uniform(g['z_min'],g['z_max'],k)
            on=np.column_stack((np.cos(theta),np.sin(theta),np.zeros(k)))
            wall=np.column_stack((g['pipe_radius']*np.cos(theta),g['pipe_radius']*np.sin(theta),z))
            caps=[];cn=[]
            for sign,zz in [(-1,g['z_min']),(1,g['z_max'])]:
                rho=g['pipe_radius']*np.sqrt(rng.random(k//2));th=rng.uniform(0,2*np.pi,k//2)
                caps.append(np.column_stack((rho*np.cos(th),rho*np.sin(th),np.full(k//2,zz))))
                cn.append(np.tile([0,0,sign],(k//2,1)))
            boundary=np.vstack((rx,wall,*caps));normals=np.vstack((-rxn,on,*cn));labels=np.array(['rx']*k+['outer']*k+['cap']*k)
    domain=Domain(interior_points=x,boundary_points=boundary,normal_vectors=normals,sample_points=x)
    domain.labels=labels
    return domain
