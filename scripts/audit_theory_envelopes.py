"""Independent small-matrix checks of the continuous SSH inequalities."""

import time

import run_theory_upgrade_hpc as study

mp, old = study.mp, study.old


def run():
    tick, rows = time.perf_counter(), []
    with mp.workdps(90):
        for length in (8, 12, 16):
            for g in (mp.mpf("-.25"), mp.mpf(0), mp.mpf(".25")):
                ctx = study.direct_context(length, g)
                c = study.constants(g)
                eta = (1-c["rho"]**2)/(1+c["rho"]**2)
                jnorm = mp.mpf("2.5")+mp.exp(abs(g))
                n, m = length//2, length//4
                ratio = [s/(2+u) for s,u in zip(ctx["singular"],ctx["mu"])]
                graph = ctx["u"]*mp.diag(ratio)*ctx["v"].T
                for i in range(n):
                    for j in range(n):
                        graph[i,j] *= mp.exp(g*(i-j))
                input_raw = mp.matrix(length,n)
                for i in range(n):
                    input_raw[2*i,i] = 1
                    for j in range(n):
                        input_raw[2*i+1,j] = graph[j,i]
                input_space = mp.qr(input_raw,mode="skinny")[0]
                alpha, deltas = {}, {}
                r = (mp.mpf("1.25")+mp.cosh(abs(g)))/4
                tail_constant = (mp.mpf(".5")+mp.exp(abs(g)))/(4*(1-r))
                for name in ("left","right"):
                    initial = old.base.q0_matrix(length,name)
                    alpha[name] = old.previous.mp_smin(input_space.T*initial)
                    occupied = list(range(m)) if name == "left" else list(range(m,n))
                    empty = [x for x in range(n) if x not in occupied]
                    block = mp.matrix([[graph[i,j] for j in occupied] for i in empty])
                    deltas[name] = old.previous.mp_smin(block)
                    tail = tail_constant*r**(m-1 if name=="left" else m)
                    assert alpha[name] <= deltas[name]+mp.mpf("1e-70") and deltas[name] <= tail
                generator = old.previous.generator(dict(length=length,gamma=2.,skin=str(g),t1=.5,t2=1.))
                for duration in (mp.mpf(".1"),mp.mpf(".5"),mp.mpf(1),mp.mpf(3)):
                    exponential = mp.expm(generator*duration)
                    u,s,vh = mp.svd(exponential)
                    w = u[:,:n]
                    ew = old.previous.mp_opnorm(ctx["perpendicular"].T*w)
                    output_bound = mp.exp(-2*c["a"]*duration)
                    singular_ratio = s[n]/s[n-1]
                    assert ew <= output_bound and singular_ratio <= output_bound
                    initial = old.base.q0_matrix(length,"charge_density_wave")
                    cdw = mp.qr(exponential*initial,mode="skinny")[0]
                    dc = old.previous.mp_opnorm(ctx["perpendicular"].T*cdw)
                    uk = c["b"]*c["rho"]/(2*c["a"])
                    kc = c["rho"]*mp.exp(-2*c["a"]*duration)/(1-uk*(1-mp.exp(-2*c["a"]*duration)))
                    cdw_bound = kc/mp.sqrt(1+kc*kc)
                    assert dc <= cdw_bound
                    for name in ("left","right"):
                        q = mp.qr(exponential*old.base.q0_matrix(length,name),mode="skinny")[0]
                        d = old.previous.mp_opnorm(ctx["perpendicular"].T*q)
                        zeta = alpha[name]/eta
                        lower = eta/(1+zeta/(1-zeta)*mp.exp(2*jnorm*duration)) if zeta < 1 else mp.mpf(0)
                        memory = old.pair_distance(q,cdw)
                        memory_lower = max(mp.mpf(0),lower-cdw_bound)
                        assert d+mp.mpf("1e-70") >= lower and memory+mp.mpf("1e-70") >= memory_lower
                        rows.append(dict(length=length,skin=str(g),time=str(duration),initial=name,
                                         alpha=str(alpha[name]),delta=str(deltas[name]),distance_to_G=str(d),delay_lower=str(lower),
                                         cdw_distance=str(dc),cdw_upper=str(cdw_bound),direct_memory=str(memory),memory_lower=str(memory_lower),
                                         output_distance=str(ew),singular_ratio=str(singular_ratio),output_upper=str(output_bound)))
    study.write_json(study.OUT/"development/theory_envelope_checks.json",
                     dict(status="passed",rows=rows, precision_digits=90, source_sha256=study.sha(study.SOURCE),
                          audit_source_sha256=study.sha(study.Path(__file__)), completed_utc=study.now(),seconds=time.perf_counter()-tick))
    study.emit(dict(stage="theory_envelope_checks",status="passed",conditions=len(rows),seconds=time.perf_counter()-tick))


if __name__=="__main__":
    run()
