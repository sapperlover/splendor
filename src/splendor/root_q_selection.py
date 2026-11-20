"""Equal-world Q selection with a per-world root visit floor."""
from __future__ import annotations

from dataclasses import dataclass, fields
import math

from .mcts_policy import MCTSSearchResult


@dataclass(frozen=True, slots=True)
class RootQSettings:
    min_visits: int = 32
    candidates: int = 8

    def __post_init__(self):
        if any(type(n) is not int or n<1 for n in (self.min_visits,self.candidates)):
            raise ValueError('root Q minimum visits and shortlist size must be positive integers')

    def initial_limit(self, budget: int, width: int) -> int:
        reserved=min(width,self.candidates)*self.min_visits
        if budget<=reserved:
            raise ValueError(f'每世界预算{budget}不足：Q补评预留{reserved}次，还需至少1次初始搜索；'
                             '请增大--simulations-per-world或降低--root-q-min-visits/--root-q-candidates')
        return budget-reserved


@dataclass(frozen=True, slots=True)
class RootQSearchResult(MCTSSearchResult):
    world_mean_q: tuple[float | None, ...] = ()
    eligible_root_candidates: tuple[int, ...] = ()
    coverage_candidates: tuple[int, ...] = ()
    root_q_min_visits: int = 0
    precoverage_root_visit_counts: tuple[tuple[int, ...], ...] = ()


def root_coverage_shortlist(visits, prior, settings: RootQSettings):
    counts=[sum(world[i] for world in visits) for i in range(len(prior))]
    ranked=sorted(range(len(prior)),key=lambda i:(-counts[i],-prior[i],i))
    return tuple(ranked[:settings.candidates])


def select_root_by_world_mean_q(result: MCTSSearchResult, settings: RootQSettings,
                                coverage_candidates, initial_visits):
    if not result.searchable:
        return result
    visits=result.per_world_root_visit_counts
    sums=result.per_world_root_value_sums
    width=len(result.root_visit_counts)
    if not visits or len(visits)!=len(sums) or any(len(row)!=width for row in (*visits,*sums)):
        raise ValueError('equal-world Q selection requires complete per-world root statistics')
    mean_q=tuple(sum(sums[w][i]/visits[w][i] for w in range(len(visits)))/len(visits)
                 if all(row[i]>0 for row in visits) else None for i in range(width))
    eligible=tuple(i for i in range(width) if all(row[i]>=settings.min_visits for row in visits))
    if not eligible or any(i not in eligible for i in coverage_candidates):
        raise ValueError('root coverage did not satisfy the per-world visit floor')
    if any(not math.isfinite(mean_q[i]) for i in eligible):
        raise ValueError('non-finite equal-world root Q')
    chosen=max(eligible,key=lambda i:(mean_q[i],result.root_visit_counts[i],result.root_network_prior[i],-i))
    base={field.name:getattr(result,field.name) for field in fields(MCTSSearchResult)}
    base.update(chosen_candidate_index=chosen,
                behavior_policy=tuple(float(i==chosen) for i in range(width)))
    return RootQSearchResult(**base,world_mean_q=mean_q,eligible_root_candidates=eligible,
        coverage_candidates=tuple(coverage_candidates),root_q_min_visits=settings.min_visits,
        precoverage_root_visit_counts=tuple(tuple(row) for row in initial_visits))
