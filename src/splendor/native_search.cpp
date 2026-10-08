#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <chrono>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <memory>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
using V6 = std::array<int, 6>;
using V5 = std::array<int, 5>;
using namespace pybind11::literals;
static constexpr uint64_t gamma64 = 0x9E3779B97F4A7C15ULL;
static const char* colors[] = {"white", "blue", "green", "red", "black", "gold"};
static const char* phases[] = {"main_action", "payment", "return_tokens", "choose_noble", "terminal"};
static const char* types[] = {"TakeDistinct", "TakePair", "ReserveMarket", "ReserveDeck", "BuyMarket", "BuyReserved", "Pay", "ReturnTokens", "ClaimNoble"};

static int color_id(const std::string& name) {
    for (int i=0;i<6;++i) if (name==colors[i]) return i;
    throw std::invalid_argument("unknown color: " + name);
}
static int phase_id(const std::string& name) {
    for (int i=0;i<5;++i) if (name==phases[i]) return i;
    throw std::invalid_argument("unknown phase: " + name);
}
static std::string enum_value(py::handle obj) { return py::str(obj.attr("value")); }
static V6 token_vector(py::handle obj) {
    V6 v{}; for (int i=0;i<6;++i) v[i] = py::cast<int>(obj.attr(colors[i])); return v;
}
static V5 gem_vector(py::handle obj) {
    V5 v{}; for (int i=0;i<5;++i) v[i] = py::cast<int>(obj.attr(colors[i])); return v;
}
static int total(const V6& a) { return std::accumulate(a.begin(), a.end(), 0); }
static uint64_t splitmix_step(uint64_t& state) {
    state += gamma64;
    uint64_t z=state;
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;
    z=(z^(z>>27))*0x94D049BB133111EBULL;
    return z^(z>>31);
}
static uint64_t randbelow(uint64_t& state, uint64_t bound) {
    const uint64_t threshold = (uint64_t(0)-bound)%bound;
    uint64_t value;
    do { value=splitmix_step(state); } while (value<threshold);
    return value%bound;
}
template <typename T> static void shuffle(std::vector<T>& v, uint64_t& state) {
    for (size_t i=v.size();i>1;--i) std::swap(v[i-1],v[randbelow(state,i)]);
}
static uint64_t derived_seed(uint64_t word, uint64_t base, uint64_t sample) {
    return word ^ base ^ ((sample+1)*gamma64);
}

struct Card { std::string id; int tier=0, bonus=0, points=0; V5 cost{}; };
struct Noble { std::string id; int points=0; V5 req{}; };
struct Catalog {
    std::vector<Card> cards;
    std::vector<Noble> nobles;
    std::unordered_map<std::string,int> card_ids, noble_ids;
    int max_reserved=3, max_tokens=10, prestige=15;
    Catalog(py::object source) {
        for (py::handle item: source.attr("cards")) {
            Card c; c.id=py::str(item.attr("card_id")); c.tier=py::cast<int>(item.attr("tier"))-1;
            c.bonus=color_id(enum_value(item.attr("bonus"))); c.points=py::cast<int>(item.attr("points")); c.cost=gem_vector(item.attr("cost"));
            card_ids.emplace(c.id, int(cards.size())); cards.push_back(c);
        }
        for (py::handle item: source.attr("nobles")) {
            Noble n; n.id=py::str(item.attr("noble_id")); n.points=py::cast<int>(item.attr("points")); n.req=gem_vector(item.attr("requirement"));
            noble_ids.emplace(n.id,int(nobles.size())); nobles.push_back(n);
        }
        if(cards.size()!=90 || nobles.size()!=10)throw std::invalid_argument("native feature schema requires 90 cards and 10 nobles");
        py::object c=source.attr("constants");
        max_reserved=py::cast<int>(c.attr("max_reserved_cards"));
        max_tokens=py::cast<int>(c.attr("max_tokens_at_turn_end"));
        prestige=py::cast<int>(c.attr("end_game_prestige"));
    }
    int card(py::handle id) const { return card_ids.at(py::cast<std::string>(id)); }
    int noble(py::handle id) const { return noble_ids.at(py::cast<std::string>(id)); }
};

struct Action {
    int type=0, id=-1, tier=-1; V6 tokens{};
    bool operator==(const Action& x) const { return type==x.type && id==x.id && tier==x.tier && tokens==x.tokens; }
};
static Action parse_action(py::handle obj, const Catalog& c) {
    std::string name=py::str(obj.get_type().attr("__name__"));
    Action a;
    for (;a.type<9;++a.type) if (name==types[a.type]) break;
    if (a.type==9) throw std::invalid_argument("unknown action type");
    if (a.type==0) for (py::handle color: obj.attr("colors")) a.tokens[color_id(enum_value(color))]=1;
    if (a.type==1) a.tokens[color_id(enum_value(obj.attr("color")))]=2;
    if (a.type==2 || a.type==4 || a.type==5) a.id=c.card(obj.attr("card_id"));
    if (a.type==3) a.tier=py::cast<int>(obj.attr("tier"))-1;
    if (a.type==6 || a.type==7) a.tokens=token_vector(obj.attr("tokens"));
    if (a.type==8) a.id=c.noble(obj.attr("noble_id"));
    return a;
}
struct Reservation { int card=-1; int source=0; };
struct Player {
    V6 tokens{}; std::vector<int> purchased, nobles; std::vector<Reservation> reserved;
};
struct State {
    V6 supply{};
    std::array<std::vector<int>,3> decks, market;
    std::array<Player,2> players;
    std::vector<int> nobles;
    int first=0, actor=0, turns[2]={0,0}, skips=0, forced_skip_total=0, phase=0, trigger=-1;
    Action pending;
    bool has_pending=false;
    int return_count=0;
    std::vector<int> eligible;
    int winner=-2; // -2 nonterminal, -1 draw, 0/1 unique winner
};
static bool card_less(int lhs,int rhs,const Catalog& c) {return c.cards[lhs].id<c.cards[rhs].id;}
static bool noble_less(int lhs,int rhs,const Catalog& c) {return c.nobles[lhs].id<c.nobles[rhs].id;}
static std::vector<int> sorted_market(const State& s,const Catalog& c) {
    std::vector<int> ids; for (auto& tier:s.market) ids.insert(ids.end(),tier.begin(),tier.end());
    std::sort(ids.begin(),ids.end(),[&](int a,int b){return card_less(a,b,c);}); return ids;
}
static V5 bonuses(const Player& p,const Catalog& c) {
    V5 b{}; for (int id:p.purchased) ++b[c.cards[id].bonus]; return b;
}
static int score(const Player& p,const Catalog& c) {
    int result=0; for (int id:p.purchased) result+=c.cards[id].points;
    for (int id:p.nobles) result+=c.nobles[id].points; return result;
}
static bool can_pay(const Player& p,const Card& card,const V5& b) {
    int shortfall=0; for(int i=0;i<5;++i) shortfall+=std::max(0,card.cost[i]-b[i]-p.tokens[i]);
    return shortfall<=p.tokens[5];
}
static std::vector<int> eligible(const State& s,const Catalog& c) {
    V5 b=bonuses(s.players[s.actor],c); std::vector<int> ids;
    for (int id:s.nobles) {
        bool ok=true; for(int i=0;i<5;++i) ok&=b[i]>=c.nobles[id].req[i];
        if(ok) ids.push_back(id);
    }
    std::sort(ids.begin(),ids.end(),[&](int a,int b){return noble_less(a,b,c);}); return ids;
}
static void payment_recurse(const Player& p,const Card& card,const V5& b,int i,int gold,V6& paid,std::vector<Action>& out) {
    if(i==5) { paid[5]=gold; Action a; a.type=6;a.tokens=paid;out.push_back(a);return; }
    int need=std::max(0,card.cost[i]-b[i]);
    for (int g=0;g<=need;++g) {
        int ordinary=need-g;
        if(ordinary>p.tokens[i] || gold+g>p.tokens[5]) continue;
        paid[i]=ordinary; payment_recurse(p,card,b,i+1,gold+g,paid,out);
    }
}
static void return_recurse(const Player& p,int i,int remaining,V6& values,std::vector<Action>& out) {
    if(i==6) {if(remaining==0){Action a;a.type=7;a.tokens=values;out.push_back(a);}return;}
    for(int v=0;v<=std::min(p.tokens[i],remaining);++v){values[i]=v;return_recurse(p,i+1,remaining-v,values,out);}
}
static std::vector<Action> actions(const State& s,const Catalog& c) {
    std::vector<Action> out;
    const Player& p=s.players[s.actor];
    if(s.phase==0) {
        std::vector<int> available;
        for(int i=0;i<5;++i) if(s.supply[i]>0) available.push_back(i);
        if(available.size()>=3) {
            for(size_t a=0;a<available.size();++a) for(size_t b=a+1;b<available.size();++b)
                for(size_t d=b+1;d<available.size();++d) {
                    Action x;x.type=0;x.tokens[available[a]]=x.tokens[available[b]]=x.tokens[available[d]]=1;out.push_back(x);
                }
        } else if(!available.empty()) {Action x;x.type=0;for(int i:available)x.tokens[i]=1;out.push_back(x);}
        for(int i=0;i<5;++i) if(s.supply[i]>=4) {Action x;x.type=1;x.tokens[i]=2;out.push_back(x);}
        auto market=sorted_market(s,c);
        if(int(p.reserved.size())<c.max_reserved) {
            for(int id:market){Action x;x.type=2;x.id=id;out.push_back(x);}
            for(int i=0;i<3;++i) if(!s.decks[i].empty()){Action x;x.type=3;x.tier=i;out.push_back(x);}
        }
        V5 b=bonuses(p,c);
        for(int id:market) if(can_pay(p,c.cards[id],b)){Action x;x.type=4;x.id=id;out.push_back(x);}
        std::vector<int> reserved;for(auto& r:p.reserved)reserved.push_back(r.card);
        std::sort(reserved.begin(),reserved.end(),[&](int a,int b){return card_less(a,b,c);});
        for(int id:reserved) if(can_pay(p,c.cards[id],b)){Action x;x.type=5;x.id=id;out.push_back(x);}
    } else if(s.phase==1) {
        const Card& card=c.cards[s.pending.id];V5 b=bonuses(p,c);V6 paid{};
        payment_recurse(p,card,b,0,0,paid,out);
        std::sort(out.begin(),out.end(),[](const Action& a,const Action& b){return a.tokens<b.tokens;});
        out.erase(std::unique(out.begin(),out.end()),out.end());
    } else if(s.phase==2) {
        V6 values{};return_recurse(p,0,total(p.tokens)-c.max_tokens,values,out);
    } else if(s.phase==3) {
        for(int id:eligible(s,c)){Action x;x.type=8;x.id=id;out.push_back(x);}
    }
    return out;
}
static void erase_id(std::vector<int>& values,int id) {
    auto it=std::find(values.begin(),values.end(),id);
    if(it==values.end()) throw std::logic_error("missing card or noble in native state");
    values.erase(it);
}
static void key_int(std::string& key,int value) {
    uint32_t v=uint32_t(value);for(int i=0;i<4;++i)key.push_back(char((v>>(8*i))&255));
}
struct ChancePool {
    int visits=0,pool_size=0;
    std::vector<int> outcomes;
};
using ChancePools=std::map<std::string,ChancePool>;
struct ChanceStream {
    uint64_t state;int draws=0;
    ChancePools* pools=nullptr;double pw_k=0,pw_alpha=.5;
    int pw_fresh=0,pw_reused=0;
};
static int draw_card(State& s,int tier,ChanceStream* chance=nullptr) {
    auto& deck=s.decks[tier];
    if(chance) {
        size_t index;
        if(chance->pools) {
            // Condition on the actual remaining pool: an opponent's resampled blind
            // card must never be drawn again. Cache only card IDs, never a successor
            // State, so private particles and deterministic rules remain current.
            auto sorted=deck;std::sort(sorted.begin(),sorted.end());std::string key;
            key_int(key,tier);for(int card:sorted)key_int(key,card);
            auto& pool=(*chance->pools)[key];++pool.visits;pool.pool_size=int(deck.size());
            double limit=std::max(1.0,std::ceil(chance->pw_k*std::pow(pool.visits,chance->pw_alpha)));
            size_t capacity=size_t(std::min(double(deck.size()),limit));
            if(pool.outcomes.size()<capacity) {
                index=randbelow(chance->state,deck.size());int card=deck[index];++chance->pw_fresh;
                if(std::find(pool.outcomes.begin(),pool.outcomes.end(),card)==pool.outcomes.end())
                    pool.outcomes.push_back(card);
            } else {
                // Every card in a Splendor pool has equal physical probability.
                // Normalize that probability on the admitted support, not visit
                // counts or value estimates (which would bias toward a lucky draw).
                int card=pool.outcomes[randbelow(chance->state,pool.outcomes.size())];
                index=size_t(std::find(deck.begin(),deck.end(),card)-deck.begin());++chance->pw_reused;
            }
        } else index=randbelow(chance->state,deck.size());
        std::swap(deck[index],deck.back());++chance->draws;
    }
    int id=deck.back();deck.pop_back();return id;
}
static void refill(State& s,int tier,ChanceStream* chance=nullptr) {
    if(!s.decks[tier].empty()) s.market[tier].push_back(draw_card(s,tier,chance));
}
static void finish(State& s,const Catalog& c) {
    int actor=s.actor;
    ++s.turns[actor];
    if(s.trigger<0 && score(s.players[actor],c)>=c.prestige) s.trigger=actor;
    s.has_pending=false;s.eligible.clear();s.return_count=0;
    if(s.trigger>=0 && s.turns[0]==s.turns[1]) {
        int a=score(s.players[0],c),b=score(s.players[1],c);
        int ac=int(s.players[0].purchased.size()),bc=int(s.players[1].purchased.size());
        s.winner=a>b?0:a<b?1:ac<bc?0:ac>bc?1:-1;s.phase=4;
    } else {s.actor=1-actor;s.phase=0;}
}
static void settle(State& s,const Catalog& c,const Action& main) {
    s.eligible=eligible(s,c);
    if(!s.eligible.empty()) {s.phase=3;s.pending=main;s.has_pending=true;}
    else finish(s,c);
}
static void apply(State& s,const Action& a,const Catalog& c,ChanceStream* chance=nullptr) {
    if(a.type<=5) s.skips=0;
    Player& p=s.players[s.actor];
    if(a.type==0 || a.type==1) {
        for(int i=0;i<6;++i){p.tokens[i]+=a.tokens[i];s.supply[i]-=a.tokens[i];}
    } else if(a.type==2) {
        int tier=c.cards[a.id].tier;erase_id(s.market[tier],a.id);
        p.reserved.push_back({a.id,0});refill(s,tier,chance);
        if(s.supply[5]){--s.supply[5];++p.tokens[5];}
    } else if(a.type==3) {
        int id=draw_card(s,a.tier,chance);p.reserved.push_back({id,1});
        if(s.supply[5]){--s.supply[5];++p.tokens[5];}
    } else if(a.type==4 || a.type==5) {
        s.pending=a;s.has_pending=true;s.phase=1;return;
    } else if(a.type==6) {
        int id=s.pending.id;
        if(s.pending.type==4){int tier=c.cards[id].tier;erase_id(s.market[tier],id);refill(s,tier,chance);}
        else {
            auto it=std::find_if(p.reserved.begin(),p.reserved.end(),[&](const Reservation& r){return r.card==id;});
            if(it==p.reserved.end()) throw std::logic_error("missing reserved purchase");
            p.reserved.erase(it);
        }
        for(int i=0;i<6;++i){p.tokens[i]-=a.tokens[i];s.supply[i]+=a.tokens[i];}
        p.purchased.push_back(id);settle(s,c,s.pending);return;
    } else if(a.type==7) {
        for(int i=0;i<6;++i){p.tokens[i]-=a.tokens[i];s.supply[i]+=a.tokens[i];}
        settle(s,c,s.pending);return;
    } else if(a.type==8) {
        p.nobles.push_back(a.id);erase_id(s.nobles,a.id);finish(s,c);return;
    }
    int excess=total(p.tokens)-c.max_tokens;
    if(excess>0){s.phase=2;s.pending=a;s.has_pending=true;s.return_count=excess;}
    else settle(s,c,a);
}
static void forced_skip(State& s,const Catalog& c) {
    int actor=s.actor;++s.turns[actor];++s.skips;++s.forced_skip_total;
    if(s.trigger>=0 && s.turns[0]==s.turns[1]) {
        int a=score(s.players[0],c),b=score(s.players[1],c);
        int ac=int(s.players[0].purchased.size()),bc=int(s.players[1].purchased.size());
        s.winner=a>b?0:a<b?1:ac<bc?0:ac>bc?1:-1;s.phase=4;
    } else if(s.skips==2) {s.winner=-1;s.phase=4;}
    else s.actor=1-actor;
}
static std::vector<Action> decide(State& s,const Catalog& c) {
    for(;;) {
        if(s.phase==4) return {};
        std::vector<Action> a=actions(s,c);
        if(s.phase==3 && a.size()==1){apply(s,a[0],c);continue;}
        if(s.phase==0 && a.empty()){forced_skip(s,c);continue;}
        if(a.empty()) throw std::logic_error("native nonterminal phase has no actions");
        return a;
    }
}
static int apply_folded(State& s,const Action& a,const Catalog& c,bool fold,int cap,std::vector<Action>& trail) {
    apply(s,a,c);trail.push_back(a);int transitions=1;
    std::vector<Action> next=decide(s,c);
    while(fold && s.phase!=4 && next.size()==1 && transitions<cap) {
        apply(s,next[0],c);trail.push_back(next[0]);++transitions;next=decide(s,c);
    }
    return transitions;
}

static std::vector<int> purchased_ids(py::handle player,const Catalog& c) {
    std::vector<int> result; for(py::handle id:player.attr("purchased_card_ids")) result.push_back(c.card(id));return result;
}
static std::vector<int> noble_ids(py::handle player,const Catalog& c) {
    std::vector<int> result; for(py::handle id:player.attr("noble_ids")) result.push_back(c.noble(id));return result;
}
static std::vector<Reservation> own_reserved(py::handle obs,const Catalog& c) {
    std::array<std::vector<int>,3> blind;
    for(py::handle card:obs.attr("self_private").attr("blind_reserved_cards")) {
        int id=c.card(card.attr("card_id"));blind[c.cards[id].tier].push_back(id);
    }
    for(int t=0;t<3;++t) std::sort(blind[t].begin(),blind[t].end(),[&](int a,int b){return card_less(a,b,c);});
    int cursor[3]={0,0,0};std::vector<Reservation> result;
    for(py::handle r:obs.attr("self_public").attr("reserved_cards")) {
        int tier=py::cast<int>(r.attr("tier"))-1;
        if(enum_value(r.attr("source"))=="market") result.push_back({c.card(r.attr("card").attr("card_id")),0});
        else result.push_back({blind[tier].at(cursor[tier]++),1});
    }
    return result;
}
static State sampled_state(py::handle obs,const Catalog& c,uint64_t seed) {
    State s;s.supply=token_vector(obs.attr("supply"));s.phase=phase_id(enum_value(obs.attr("phase")));
    s.first=py::cast<bool>(obs.attr("self_is_first_player"))?0:1;
    s.turns[0]=py::cast<int>(obs.attr("own_completed_turns"));s.turns[1]=py::cast<int>(obs.attr("opponent_completed_turns"));
    s.skips=py::cast<int>(obs.attr("consecutive_forced_skips"));
    if(py::cast<bool>(obs.attr("final_round_triggered"))) s.trigger=py::cast<bool>(obs.attr("final_round_triggered_by_self"))?0:1;
    for(int t=0;t<3;++t) {
        py::handle view=obs.attr("market").attr(t==0?"tier_one":t==1?"tier_two":"tier_three");
        for(py::handle card:view) s.market[t].push_back(c.card(card.attr("card_id")));
    }
    s.players[0].tokens=token_vector(obs.attr("self_public").attr("tokens"));
    s.players[1].tokens=token_vector(obs.attr("opponent_public").attr("tokens"));
    s.players[0].purchased=purchased_ids(obs.attr("self_public"),c);
    s.players[1].purchased=purchased_ids(obs.attr("opponent_public"),c);
    s.players[0].nobles=noble_ids(obs.attr("self_public"),c);
    s.players[1].nobles=noble_ids(obs.attr("opponent_public"),c);
    s.players[0].reserved=own_reserved(obs,c);
    std::vector<bool> visible(c.cards.size());
    for(auto& tier:s.market) for(int id:tier) visible[id]=true;
    for(auto& p:s.players) for(int id:p.purchased) visible[id]=true;
    for(auto& r:s.players[0].reserved) visible[r.card]=true;
    for(py::handle r:obs.attr("opponent_public").attr("reserved_cards"))
        if(enum_value(r.attr("source"))=="market") visible[c.card(r.attr("card").attr("card_id"))]=true;
    std::array<std::vector<int>,3> blind;
    uint64_t state=seed;
    for(int t=0;t<3;++t) {
        std::vector<int> unknown;
        for(size_t id=0;id<c.cards.size();++id) if(c.cards[id].tier==t && !visible[id]) unknown.push_back(int(id));
        std::sort(unknown.begin(),unknown.end(),[&](int a,int b){return card_less(a,b,c);});
        shuffle(unknown,state);
        int blind_count=0;
        for(py::handle r:obs.attr("opponent_public").attr("reserved_cards"))
            if(enum_value(r.attr("source"))=="deck" && py::cast<int>(r.attr("tier"))==t+1) ++blind_count;
        int deck_count=py::cast<int>(py::reinterpret_borrow<py::tuple>(obs.attr("deck_counts"))[t]);
        if(int(unknown.size())!=blind_count+deck_count) throw std::invalid_argument("hidden card count mismatch");
        blind[t]=std::vector<int>(unknown.begin(),unknown.begin()+blind_count);
        s.decks[t]=std::vector<int>(unknown.begin()+blind_count,unknown.end());
    }
    int cursor[3]={0,0,0};
    for(py::handle r:obs.attr("opponent_public").attr("reserved_cards")) {
        int tier=py::cast<int>(r.attr("tier"))-1;
        if(enum_value(r.attr("source"))=="market") s.players[1].reserved.push_back({c.card(r.attr("card").attr("card_id")),0});
        else s.players[1].reserved.push_back({blind[tier].at(cursor[tier]++),1});
    }
    for(py::handle noble:obs.attr("nobles_in_play")) s.nobles.push_back(c.noble(noble.attr("noble_id")));
    if(!obs.attr("pending").is_none()) {
        py::handle pending=obs.attr("pending");s.has_pending=true;s.pending=parse_action(pending.attr("main_action"),c);
        s.return_count=py::cast<int>(pending.attr("tokens_to_return"));
        for(py::handle noble:pending.attr("eligible_nobles")) s.eligible.push_back(c.noble(noble.attr("noble_id")));
    }
    return s;
}
static constexpr int CARD_DIM=29, NOBLE_DIM=17, GLOBAL_DIM=148, ACTION_DIM=82;
static const double token_scale[6]={4,4,4,4,4,5};
static const double deck_scale[3]={40,30,20};
struct CardRow {int id=-1,tier=-1,source=-1;};
static std::array<float,CARD_DIM> card_features(const CardRow& row,const State& s,const Catalog& c,int observer) {
    std::array<float,CARD_DIM> f{};
    if(row.id<0 && row.tier<0 && row.source<0) return f;
    f[0]=1;
    if(row.id>=0)f[1]=1;
    if(row.tier>=0)f[2+row.tier]=1;
    if(row.source>=0)f[10+row.source]=1;
    if(row.id<0)return f;
    const Card& card=c.cards[row.id];const Player& player=s.players[observer];V5 b=bonuses(player,c);
    f[5+card.bonus]=1;f[12]=float(card.points/5.0);
    int deficit_sum=0;
    for(int i=0;i<5;++i) {
        int need=std::max(0,card.cost[i]-b[i]);int deficit=std::max(0,need-player.tokens[i]);
        f[13+i]=float(card.cost[i]/7.0);f[18+i]=float(need/7.0);f[23+i]=float(deficit/7.0);
        deficit_sum+=deficit;
    }
    f[28]=float(std::max(0,deficit_sum-player.tokens[5])/14.0);return f;
}
static std::array<float,NOBLE_DIM> noble_features(int id,const State& s,const Catalog& c,int observer) {
    std::array<float,NOBLE_DIM> f{};
    const Noble& n=c.nobles[id];V5 own=bonuses(s.players[observer],c),opp=bonuses(s.players[1-observer],c);
    f[0]=1;f[1]=float(n.points/3.0);
    for(int i=0;i<5;++i) {
        f[2+i]=float(n.req[i]/4.0);f[7+i]=float(std::max(0,n.req[i]-own[i])/4.0);
        f[12+i]=float(std::max(0,n.req[i]-opp[i])/4.0);
    }
    return f;
}
static bool reserved_less(const Reservation& a,const Reservation& b,const Catalog& c) {
    // Python sorts ReservationSource.value: "deck" before "market".
    if(a.source!=b.source)return a.source>b.source;
    int at=c.cards[a.card].tier,bt=c.cards[b.card].tier;
    if(at!=bt)return at<bt;
    if(a.source==1)return false;
    return card_less(a.card,b.card,c);
}
static std::vector<CardRow> reservation_rows(const State& s,const Catalog& c,int owner,int observer) {
    std::vector<CardRow> rows;
    auto rs=s.players[owner].reserved;
    std::stable_sort(rs.begin(),rs.end(),[&](const Reservation& a,const Reservation& b){return reserved_less(a,b,c);});
    if(owner==observer) {
        for(auto& r:rs) if(r.source==0)rows.push_back({r.card,c.cards[r.card].tier,0});
        std::vector<int> blind;for(auto& r:rs)if(r.source==1)blind.push_back(r.card);
        std::sort(blind.begin(),blind.end(),[&](int a,int b){return card_less(a,b,c);});
        for(int id:blind)rows.push_back({id,c.cards[id].tier,1});
    } else {
        for(auto& r:rs)rows.push_back({r.source==0?r.card:-1,c.cards[r.card].tier,r.source});
    }
    return rows;
}
static std::vector<int> sorted_nobles(const std::vector<int>& input,const Catalog& c) {
    std::vector<int> out=input;std::sort(out.begin(),out.end(),[&](int a,int b){return noble_less(a,b,c);});return out;
}
static std::array<float,GLOBAL_DIM> global_features(const State& s,const Catalog& c,int observer) {
    std::array<float,GLOBAL_DIM> f{};int k=0;
    for(int i=0;i<6;++i)f[k++]=float(s.supply[i]/token_scale[i]);
    for(int t=0;t<3;++t)f[k++]=float(s.decks[t].size()/deck_scale[t]);
    for(int player:{observer,1-observer}) {
        const Player& p=s.players[player];V5 b=bonuses(p,c);
        for(int i=0;i<6;++i)f[k++]=float(p.tokens[i]/token_scale[i]);
        for(int i=0;i<5;++i)f[k++]=float(b[i]/18.0);
        f[k++]=float(score(p,c)/15.0);f[k++]=float(p.purchased.size()/90.0);f[k++]=float(p.nobles.size()/3.0);
    }
    std::vector<bool> bought(c.cards.size());for(auto& p:s.players)for(int id:p.purchased)bought[id]=true;
    for(size_t i=0;i<c.cards.size();++i)f[k++]=bought[i]?1:0;
    f[k++]=observer==s.first?1:0;
    for(int p=0;p<4;++p)f[k++]=s.phase==p?1:0;
    f[k++]=s.trigger>=0?1:0;f[k++]=s.trigger==observer?1:0;
    f[k++]=float(s.turns[observer]/40.0);f[k++]=float(s.turns[1-observer]/40.0);
    f[k++]=float(s.turns[observer]-s.turns[1-observer]);f[k++]=s.turns[observer]==s.turns[1-observer]?1:0;
    f[k++]=float(s.skips/2.0);f[k++]=s.has_pending?1:0;
    for(int t=0;t<6;++t)f[k++]=s.has_pending && s.pending.type==t?1:0;
    f[k++]=s.has_pending?float(s.return_count/3.0):0;
    f[k++]=s.has_pending?float(s.eligible.size()/3.0):0;
    if(k!=GLOBAL_DIM)throw std::logic_error("global feature size mismatch");return f;
}
static CardRow target_card(const State& s,const Catalog& c,int observer,const Action& a) {
    if(a.type==2 || a.type==4)return {a.id,c.cards[a.id].tier,0};
    if(a.type==5 || a.type==6) {
        int id=a.type==5?a.id:s.pending.id;
        int source=0;
        if(a.type==5 || s.pending.type==5) {
            auto& r=s.players[observer].reserved;
            auto it=std::find_if(r.begin(),r.end(),[&](const Reservation& x){return x.card==id;});
            if(it==r.end())throw std::logic_error("reserved card target missing");
            source=it->source;
        }
        return {id,c.cards[id].tier,source};
    }
    return {};
}
static std::array<float,ACTION_DIM> action_features(const State& s,const Catalog& c,int observer,const Action& a) {
    std::array<float,ACTION_DIM> f{};
    f[s.phase]=1;f[4+a.type]=1;
    int source=-1,tier=-1;
    if(a.type==2 || a.type==4){source=0;tier=c.cards[a.id].tier;}
    if(a.type==3){source=1;tier=a.tier;}
    if(a.type==5 || a.type==6){CardRow row=target_card(s,c,observer,a);source=row.source;tier=row.tier;}
    if(source>=0)f[13+source]=1;if(tier>=0)f[15+tier]=1;
    int offset=a.type<=1?18:a.type==6?24:a.type==7?30:-1;
    if(offset>=0)for(int i=0;i<6;++i)f[offset+i]=float(a.tokens[i]/token_scale[i]);
    if(a.type==2 || a.type==4 || a.type==5 || a.type==6) {
        auto card=card_features(target_card(s,c,observer,a),s,c,observer);
        std::copy(card.begin(),card.end(),f.begin()+36);
    }
    if(a.type==8) {auto n=noble_features(a.id,s,c,observer);std::copy(n.begin(),n.end(),f.begin()+65);}
    return f;
}
template <size_t N> static py::array_t<float> array_from(const std::array<float,N>& a) {
    py::array_t<float> out(N);std::copy(a.begin(),a.end(),out.mutable_data());return out;
}
static py::array_t<float> card_matrix(const std::vector<CardRow>& rows,int cap,const State& s,const Catalog& c,int observer) {
    if(int(rows.size())>cap)throw std::invalid_argument("native card row exceeds feature capacity");
    py::array_t<float> out({cap,CARD_DIM});std::fill(out.mutable_data(),out.mutable_data()+cap*CARD_DIM,0);
    for(size_t i=0;i<rows.size();++i){auto f=card_features(rows[i],s,c,observer);std::copy(f.begin(),f.end(),out.mutable_data()+i*CARD_DIM);}
    return out;
}
static py::array_t<bool> mask_array(int cap,int count) {
    py::array_t<bool> out(cap);for(int i=0;i<cap;++i)out.mutable_data()[i]=i<count;return out;
}
static py::dict encode(const State& s,const std::vector<Action>& acts,const Catalog& c,bool state_only=false) {
    int observer=s.actor;py::dict out;
    out["feature_schema_version"]="splendor-observation-action-v1";
    out["global_features"]=array_from(global_features(s,c,observer));
    for(int t=0;t<3;++t) {
        std::vector<int> ids=s.market[t];std::sort(ids.begin(),ids.end(),[&](int a,int b){return card_less(a,b,c);});
        std::vector<CardRow> rows;for(int id:ids)rows.push_back({id,t,0});
        std::string key="market_t"+std::to_string(t+1);
        out[py::str(key)]=card_matrix(rows,4,s,c,observer);out[py::str(key+"_mask")]=mask_array(4,int(rows.size()));
    }
    auto own=reservation_rows(s,c,observer,observer),opp=reservation_rows(s,c,1-observer,observer);
    out["self_reserved"]=card_matrix(own,3,s,c,observer);out["self_reserved_mask"]=mask_array(3,int(own.size()));
    out["opponent_reserved"]=card_matrix(opp,3,s,c,observer);out["opponent_reserved_mask"]=mask_array(3,int(opp.size()));
    auto nobles=sorted_nobles(s.nobles,c);
    if(nobles.size()>3)throw std::invalid_argument("native noble row exceeds feature capacity");
    py::array_t<float> noble_array({3,NOBLE_DIM});
    std::fill(noble_array.mutable_data(),noble_array.mutable_data()+3*NOBLE_DIM,0);
    for(size_t i=0;i<nobles.size();++i){auto f=noble_features(nobles[i],s,c,observer);std::copy(f.begin(),f.end(),noble_array.mutable_data()+i*NOBLE_DIM);}
    out["nobles"]=noble_array;out["nobles_mask"]=mask_array(3,int(nobles.size()));
    bool pending=s.phase==1;
    CardRow pending_row;
    if(pending){Action payment;payment.type=6;pending_row=target_card(s,c,observer,payment);}
    out["pending_card"]=array_from(card_features(pending_row,s,c,observer));
    py::array_t<bool> pending_mask(py::array::ShapeContainer{});pending_mask.mutable_data()[0]=pending;out["pending_card_mask"]=pending_mask;
    if(!state_only) {
        py::array_t<float> matrix({int(acts.size()),ACTION_DIM});
        for(size_t i=0;i<acts.size();++i){auto f=action_features(s,c,observer,acts[i]);std::copy(f.begin(),f.end(),matrix.mutable_data()+i*ACTION_DIM);}
        out["candidate_features"]=matrix;out["candidate_mask"]=mask_array(int(acts.size()),int(acts.size()));
        out["candidate_actions"]=py::tuple(acts.size());
    }
    return out;
}
struct Edge {
    int visits=0,child=-1,terminal=-2;double value_sum=0;
    std::vector<Action> branch;
    // Sampled outcomes aggregate into this decision edge's N/W, never maximize over chance.
    std::unique_ptr<std::unordered_map<std::string,int>> outcomes;
    std::unique_ptr<ChancePools> draw_pools;
};
struct Node {
    State state;std::vector<Action> actions;std::vector<double> prior;std::vector<Edge> edges;
    Node(State s,std::vector<Action> a,std::vector<double> p):state(std::move(s)),actions(std::move(a)),prior(std::move(p)),edges(actions.size()){}
};
struct Leaf {
    State state;std::vector<Action> actions;
    std::vector<std::pair<int,int>> path;
    std::vector<Action> branch;
    int transitions=0,simulation=0;bool cap=false;
    bool exported=false; // claimed by a batch row until that row is submitted
    std::string information_key;
};
struct World {
    int budget=0,completed=0;
    std::vector<Node> nodes;
    bool has_leaf=false;Leaf leaf;
    std::unordered_map<std::string,int> information_nodes;
};

// Exact, observer-safe signatures (not floating model features or full hidden states).
// Histories include each player's own private observations and public actions, retaining recall.
template<typename Values> static void key_values(std::string& key,const Values& values) {
    key_int(key,int(values.size()));for(int value:values)key_int(key,value);
}
static void key_sorted(std::string& key,std::vector<int> values) {
    std::sort(values.begin(),values.end());key_values(key,values);
}
static bool private_purchase(const State& s,const Action& a,int observer) {
    if(a.type!=5 || observer==s.actor)return false;
    auto& reserved=s.players[s.actor].reserved;
    auto it=std::find_if(reserved.begin(),reserved.end(),[&](const Reservation& r){return r.card==a.id;});
    return it!=reserved.end() && it->source==1;
}
static void key_action(std::string& key,const State& s,const Action& a,int observer,const Catalog& c) {
    bool hidden=private_purchase(s,a,observer);
    key_int(key,s.actor);key_int(key,a.type);key_int(key,hidden?-1:a.id);
    key_int(key,a.tier);key_values(key,a.tokens);
}
static std::string information_signature(const State& s,const Catalog& c,int observer) {
    std::string key;key.reserve(512);
    for(int value:{observer,s.actor,s.first,s.phase,s.turns[0],s.turns[1],s.skips,s.trigger,s.winner})key_int(key,value);
    key_values(key,s.supply);
    for(int t=0;t<3;++t){key_int(key,int(s.decks[t].size()));key_sorted(key,s.market[t]);}
    for(int player=0;player<2;++player) {
        auto& p=s.players[player];key_values(key,p.tokens);key_sorted(key,p.purchased);key_sorted(key,p.nobles);
        std::vector<int> reservations;
        for(auto& r:p.reserved) {
            // Blind opponents' identities are absent, including their ordering.
            int visible=(player==observer || r.source==0)?r.card:-1;
            reservations.push_back((c.cards[r.card].tier+1)*1000+r.source*100+visible+1);
        }
        key_sorted(key,reservations);
    }
    key_sorted(key,s.nobles);key_int(key,s.has_pending);key_int(key,s.return_count);key_sorted(key,s.eligible);
    if(s.has_pending)key_action(key,s,s.pending,observer,c);
    return key;
}
static void append_history(std::array<std::string,2>& histories,const State& before,const Action& a,
                           const State& after,const Catalog& c) {
    for(int observer=0;observer<2;++observer) {
        key_action(histories[observer],before,a,observer,c);
        histories[observer]+=information_signature(after,c,observer);
    }
}
static State sample_hidden_root(const State& root,const Catalog& c,uint64_t seed) {
    State s=root;uint64_t rng=seed;
    for(int tier=0;tier<3;++tier) {
        std::vector<int> pool=root.decks[tier];std::vector<int> slots;
        for(size_t i=0;i<root.players[1].reserved.size();++i) {
            auto& r=root.players[1].reserved[i];
            if(r.source==1 && c.cards[r.card].tier==tier){pool.push_back(r.card);slots.push_back(int(i));}
        }
        std::sort(pool.begin(),pool.end());
        for(int slot:slots) {
            size_t index=randbelow(rng,pool.size());
            s.players[1].reserved[slot].card=pool[index];
            std::swap(pool[index],pool.back());pool.pop_back();
        }
        // An unordered remaining pool, not a fixed future deck order.
        std::sort(pool.begin(),pool.end());s.decks[tier]=std::move(pool);
    }
    return s;
}
static int apply_chance_folded(State& s,const Action& first,const Catalog& c,bool fold,int cap,
                              std::vector<Action>& trail,std::array<std::string,2>& histories,ChanceStream& chance) {
    Action action=first;int transitions=0;
    do {
        State before=s;apply(s,action,c,&chance);trail.push_back(action);++transitions;
        auto next=decide(s,c);append_history(histories,before,action,s,c);
        if(!fold || s.phase==4 || next.size()!=1 || transitions>=cap)break;
        action=next[0];
    } while(true);
    return transitions;
}
static py::dict state_snapshot(const State& state);
struct SearchConfig {
    int simulations=256,determinizations=2,max_transitions=512;
    double c_puct=1.5,root_noise_fraction=.25,dirichlet_alpha=.3;
    bool fold=true;
    SearchConfig(py::object config) {
        simulations=py::cast<int>(config.attr("simulations"));
        determinizations=py::cast<int>(config.attr("determinizations"));
        max_transitions=py::cast<int>(config.attr("max_simulation_transitions"));
        c_puct=py::cast<double>(config.attr("c_puct"));
        root_noise_fraction=py::cast<double>(config.attr("root_noise_fraction"));
        dirichlet_alpha=py::cast<double>(config.attr("dirichlet_alpha"));
        fold=py::cast<bool>(config.attr("fold_forced_phases"));
    }
};
class SearchJob {
    const Catalog& catalog;
    SearchConfig cfg;
    State root_state;
    std::vector<Action> root_actions;
    std::vector<World> worlds;
    std::vector<double> network_prior,used_prior;
    uint64_t public_word,base_seed,sampling_seed;
    bool noise,root_pending=true,finished=false,root_exported=false;
    bool stochastic=false;
    double chance_pw_k=0,chance_pw_alpha=.5;
    int root_samples=0,chance_draws=0,information_hits=0;
    int chance_pw_fresh=0,chance_pw_reused=0;
    double temperature;
    int max_depth=0,cap_hits=0,expanded=0,terminal_hits=0,transitions=0,folded=0,model_positions=0;
    double rule_seconds=0,feature_seconds=0,compute_seconds=0;
    long long tree_depth_sum=0;
    int chosen=0,current_world=-1,current_simulation=-1;
    std::vector<Action> current_branch;Action current_attempted;bool has_attempted=false;
    std::vector<double> target,behavior;
    std::vector<std::vector<int>> rows;
    std::vector<int> counts,budgets;
    static std::vector<double> softmax(const std::vector<double>& logits) {
        if(logits.empty())throw std::invalid_argument("empty network logits");
        for(double v:logits)if(!std::isfinite(v))throw std::invalid_argument("nonfinite network logits");
        double peak=*std::max_element(logits.begin(),logits.end()),sum=0;
        std::vector<double> out;for(double v:logits){out.push_back(std::exp(v-peak));sum+=out.back();}
        for(double& v:out)v/=sum;return out;
    }
    void backup(World& w,const std::vector<std::pair<int,int>>& path,double value) {
        for(auto [node,edge]:path){Edge& e=w.nodes[node].edges[edge];++e.visits;e.value_sum+=value;}
        ++w.completed;tree_depth_sum+=path.size();max_depth=std::max(max_depth,int(path.size()));
    }
    int select(const Node& n,int root_actor) const {
        int total=0;for(auto& e:n.edges)total+=e.visits;
        int sign=n.state.actor==root_actor?1:-1;
        int best=0;double best_score=-1e300,best_prior=-1;
        double factor=cfg.c_puct*std::sqrt(1+total);
        for(size_t i=0;i<n.edges.size();++i) {
            const Edge& e=n.edges[i];double q=e.visits?e.value_sum/e.visits:0;
            double s=sign*q+factor*n.prior[i]/(1+e.visits);
            if(s>best_score || (s==best_score && n.prior[i]>best_prior)) {
                best=int(i);best_score=s;best_prior=n.prior[i];
            }
        }
        return best;
    }
    void advance(int index) {
        if(stochastic){advance_chance(index);return;}
        auto compute_start=std::chrono::steady_clock::now();
        World& w=worlds[index];int root_actor=w.nodes[0].state.actor;
        while(w.completed<w.budget && !w.has_leaf) {
            current_world=index;current_simulation=w.completed;current_branch.clear();has_attempted=false;
            std::vector<std::pair<int,int>> path;
            int node_index=0;std::vector<Action> branch;
            for(;;) {
                // No node or edge reference survives a node-pool insertion.
                int edge_index=select(w.nodes[node_index],root_actor);
                path.emplace_back(node_index,edge_index);
                Edge& e=w.nodes[node_index].edges[edge_index];
                if(e.child>=0 || e.terminal>=-1) {
                    branch.insert(branch.end(),e.branch.begin(),e.branch.end());
                    current_branch=branch;
                }
                if(e.terminal>=-1){++terminal_hits;backup(w,path,e.terminal==root_actor?1:e.terminal<0?0:-1);break;}
                if(e.child>=0){node_index=e.child;continue;}
                State child=w.nodes[node_index].state;
                std::vector<Action> trail;
                current_attempted=w.nodes[node_index].actions[edge_index];has_attempted=true;
                auto rule_start=std::chrono::steady_clock::now();
                int steps=apply_folded(child,w.nodes[node_index].actions[edge_index],catalog,cfg.fold,cfg.max_transitions,trail);
                rule_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-rule_start).count();
                has_attempted=false;
                transitions+=steps;folded+=steps-1;
                w.nodes[node_index].edges[edge_index].branch=trail;
                branch.insert(branch.end(),trail.begin(),trail.end());current_branch=branch;
                if(child.phase==4) {
                    w.nodes[node_index].edges[edge_index].terminal=child.winner;
                    ++terminal_hits;backup(w,path,child.winner==root_actor?1:child.winner<0?0:-1);break;
                }
                bool cap=steps>=cfg.max_transitions;if(cap)++cap_hits;
                w.leaf=Leaf{std::move(child),{},std::move(path),std::move(branch),steps,w.completed,cap};
                if(!cap)w.leaf.actions=actions(w.leaf.state,catalog);
                w.has_leaf=true;++model_positions;break;
            }
        }
        if(!root_pending && std::all_of(worlds.begin(),worlds.end(),[](const World& x){return x.completed==x.budget && !x.has_leaf;})) finish();
        compute_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-compute_start).count();
    }
    void advance_chance(int index) {
        auto started=std::chrono::steady_clock::now();
        World& w=worlds[index];int root_actor=root_state.actor;
        while(w.completed<w.budget && !w.has_leaf) {
            current_world=index;current_simulation=w.completed;current_branch.clear();has_attempted=false;
            State state=sample_hidden_root(root_state,catalog,derived_seed(public_word,base_seed,w.completed));
            ++root_samples;
            ChanceStream chance{derived_seed(public_word,base_seed^0x4348414E4345ULL,w.completed)};
            std::array<std::string,2> histories={information_signature(state,catalog,0),information_signature(state,catalog,1)};
            std::vector<std::pair<int,int>> path;std::vector<Action> branch;
            int node_index=0,steps=0;
            for(;;) {
                int edge_index=select(w.nodes[node_index],root_actor);
                path.emplace_back(node_index,edge_index);
                Action action=w.nodes[node_index].actions[edge_index];int observer=state.actor;
                if(chance_pw_k>0) {
                    auto& pools=w.nodes[node_index].edges[edge_index].draw_pools;
                    if(!pools)pools=std::make_unique<ChancePools>();
                    chance.pools=pools.get();chance.pw_k=chance_pw_k;chance.pw_alpha=chance_pw_alpha;
                }
                current_attempted=action;has_attempted=true;
                auto rule_start=std::chrono::steady_clock::now();std::vector<Action> trail;
                int before_draws=chance.draws;
                int before_fresh=chance.pw_fresh,before_reused=chance.pw_reused;
                int applied=apply_chance_folded(state,action,catalog,cfg.fold,cfg.max_transitions-steps,trail,histories,chance);
                steps+=applied;chance_draws+=chance.draws-before_draws;
                chance_pw_fresh+=chance.pw_fresh-before_fresh;chance_pw_reused+=chance.pw_reused-before_reused;
                rule_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-rule_start).count();
                has_attempted=false;transitions+=applied;folded+=applied-1;
                branch.insert(branch.end(),trail.begin(),trail.end());current_branch=branch;
                // Revisit the chance event on every simulation, sharing only decision statistics.
                auto& outcomes=w.nodes[node_index].edges[edge_index].outcomes;
                if(!outcomes)outcomes=std::make_unique<std::unordered_map<std::string,int>>();
                ++(*outcomes)[information_signature(state,catalog,observer)];
                if(state.phase==4) {
                    ++terminal_hits;backup(w,path,state.winner==root_actor?1:state.winner<0?0:-1);break;
                }
                bool cap=steps>=cfg.max_transitions;if(cap)++cap_hits;
                auto next=actions(state,catalog);
                std::string key=histories[state.actor];auto found=w.information_nodes.find(key);
                if(!cap && found!=w.information_nodes.end()) {
                    node_index=found->second;++information_hits;
                    if(w.nodes[node_index].actions!=next)throw std::logic_error("information set legal actions differ");
                    continue;
                }
                w.leaf=Leaf{std::move(state),cap?std::vector<Action>{}:std::move(next),std::move(path),
                            std::move(branch),steps,w.completed,cap};
                w.leaf.information_key=std::move(key);w.has_leaf=true;++model_positions;break;
            }
        }
        if(!root_pending && w.completed==w.budget && !w.has_leaf)finish();
        compute_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    }
    void finish() {
        if(finished)return;
        int width=int(root_actions.size());rows.clear();counts.assign(width,0);target.assign(width,0);
        for(auto& w:worlds) {
            std::vector<int> row;
            for(int i=0;i<width;++i){int v=w.nodes[0].edges[i].visits;row.push_back(v);counts[i]+=v;target[i]+=double(v)/w.budget/worlds.size();}
            rows.push_back(row);
        }
        behavior=target;
        if(temperature==0) {
            chosen=0;for(int i=1;i<width;++i)if(target[i]>target[chosen] || (target[i]==target[chosen] && network_prior[i]>network_prior[chosen]))chosen=i;
            std::fill(behavior.begin(),behavior.end(),0);behavior[chosen]=1;
        } else {
            double peak=*std::max_element(target.begin(),target.end());double sum=0;
            for(int i=0;i<width;++i){behavior[i]=target[i]>0?std::exp((std::log(target[i])-std::log(peak))/temperature):0;sum+=behavior[i];}
            for(double& x:behavior)x/=sum;
            uint64_t state=derived_seed(public_word,sampling_seed,0x53414D50);
            double draw=(splitmix_step(state)>>11)*0x1.0p-53;
            double cumulative=0;chosen=width-1;
            for(int i=0;i<width;++i){cumulative+=behavior[i];if(draw<cumulative){chosen=i;break;}}
        }
        finished=true;
    }
public:
    SearchJob(const Catalog& catalog_,py::object request,py::object config,uint64_t word,uint64_t seed,bool noise_,double temp,bool stochastic_=false,
              double chance_pw_k_=0,double chance_pw_alpha_=.5):
        catalog(catalog_),cfg(config),public_word(word),base_seed(PyLong_AsUnsignedLongLongMask(config.attr("seed").ptr())),
        sampling_seed(seed),noise(noise_),stochastic(stochastic_),chance_pw_k(chance_pw_k_),chance_pw_alpha(chance_pw_alpha_),temperature(temp) {
        if(stochastic && cfg.determinizations!=1)throw std::invalid_argument("sampled chance search requires determinizations=1 (one shared statistics pool)");
        if(!std::isfinite(chance_pw_k) || chance_pw_k<0 || !std::isfinite(chance_pw_alpha) || chance_pw_alpha<=0 || chance_pw_alpha>=1)
            throw std::invalid_argument("chance widening requires finite k>=0 and 0<alpha<1");
        if(chance_pw_k>0 && !stochastic)throw std::invalid_argument("chance widening requires stochastic search");
        py::object obs=request.attr("observation");
        for(py::handle a:request.attr("legal_actions"))root_actions.push_back(parse_action(a,catalog));
        if(root_actions.empty())throw std::invalid_argument("root needs legal actions");
        if(root_actions.size()==1){root_pending=false;finished=true;network_prior={1};used_prior={1};counts={0};target={1};behavior={1};return;}
        root_state=sampled_state(obs,catalog,derived_seed(public_word,base_seed,0));
        State verify=root_state;auto actual=decide(verify,catalog);
        if(actual!=root_actions)throw std::invalid_argument("native root candidate order differs from Python reference");
        ++model_positions;
    }
    bool done() const{return finished;}
    bool is_root_pending() const{return root_pending && !finished;}
    int pending_count() const {int n=root_pending?1:0;for(auto& w:worlds)n+=w.has_leaf;return n;}
    py::list pending() {
        auto feature_start=std::chrono::steady_clock::now();
        py::list result;
        if(is_root_pending()){
            py::dict item;item["world"]=-1;item["kind"]="policy_value";
            item["encoded"]=encode(root_state,root_actions,catalog);
            result.append(item);
            feature_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-feature_start).count();
            return result;
        }
        for(size_t i=0;i<worlds.size();++i)if(worlds[i].has_leaf){
            auto& leaf=worlds[i].leaf;py::dict item;item["world"]=int(i);
            item["kind"]=leaf.cap?"value_only":"policy_value";
            item["encoded"]=encode(leaf.state,leaf.actions,catalog,leaf.cap);
            result.append(item);
        }
        feature_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-feature_start).count();
        return result;
    }
    void submit(int world,const std::vector<double>& logits,double value) {
        if(!std::isfinite(value))throw std::invalid_argument("nonfinite network value");
        value=std::max(-1.0,std::min(1.0,value));
        if(world==-1) {
            if(!root_pending)throw std::logic_error("root request already resolved");
            if(logits.size()!=root_actions.size())throw std::invalid_argument("root logits width mismatch");
            network_prior=softmax(logits);used_prior=network_prior;
            if(noise) {
                std::mt19937_64 rng(derived_seed(public_word,sampling_seed,0x4E4F4953));
                std::vector<double> eta(network_prior.size());double sum=0;
                if(cfg.dirichlet_alpha>1e306) {
                    std::fill(eta.begin(),eta.end(),1.0);sum=double(eta.size());
                } else {
                    std::gamma_distribution<double> dist(cfg.dirichlet_alpha,1.0);
                    for(double& v:eta){v=dist(rng);sum+=v;}
                    if(sum==0){eta[rng()%eta.size()]=1;sum=1;}
                }
                for(size_t i=0;i<eta.size();++i)used_prior[i]=(1-cfg.root_noise_fraction)*network_prior[i]+cfg.root_noise_fraction*eta[i]/sum;
            }
            int n=cfg.determinizations;budgets.resize(n);
            for(int j=0;j<n;++j) {
                int budget=cfg.simulations/n+(j<cfg.simulations%n);budgets[j]=budget;
                // All worlds are reconstructed from the same public view in the constructor's next call.
            }
            root_pending=false;
            return;
        }
        World& w=worlds.at(world);
        if(!w.has_leaf)throw std::logic_error("world has no pending leaf");
        if(!w.leaf.cap && logits.size()!=w.leaf.actions.size())throw std::invalid_argument("leaf logits width mismatch");
        if(w.leaf.cap && !logits.empty())throw std::invalid_argument("value-only leaf cannot receive logits");
        std::vector<double> prior;if(!w.leaf.cap)prior=softmax(logits);
        Leaf leaf=std::move(w.leaf);w.has_leaf=false;
        if(!leaf.cap) {
            int child=int(w.nodes.size());w.nodes.emplace_back(std::move(leaf.state),std::move(leaf.actions),std::move(prior));
            if(stochastic)w.information_nodes.emplace(std::move(leaf.information_key),child);
            else {auto [parent,edge]=leaf.path.back();w.nodes[parent].edges[edge].child=child;}
            ++expanded;
            int root_actor=w.nodes[0].state.actor;
            double v=w.nodes[child].state.actor==root_actor?value:-value;
            backup(w,leaf.path,v);
        } else {
            int root_actor=w.nodes[0].state.actor;
            backup(w,leaf.path,leaf.state.actor==root_actor?value:-value);
        }
        py::gil_scoped_release release;advance(world);
    }
    // -- batch export/import -------------------------------------------------
    // The batch interface exposes the pending leaves of one job so that several
    // jobs can fill one contiguous buffer, and consumes results in the same
    // stable order.  Row identity is (world, leaf) captured at fill time.
    bool root_pending_now() const{return root_pending&&!finished;}
    bool root_exportable() const{return root_pending && !finished && !root_exported;}
    void mark_root_exported(){root_exported=true;}
    const State& root_state_ref() const{return root_state;}
    const std::vector<Action>& root_actions_ref() const{return root_actions;}
    int world_count() const{return int(worlds.size());}
    bool world_has_leaf(int world) const{return worlds.at(world).has_leaf;}
    bool world_leaf_exportable(int world) const{
        const World& w=worlds.at(world);return w.has_leaf && !w.leaf.exported;
    }
    void mark_leaf_exported(int world){worlds.at(world).leaf.exported=true;}
    bool world_leaf_cap(int world) const{return worlds.at(world).leaf.cap;}
    int world_leaf_simulation(int world) const{return worlds.at(world).leaf.simulation;}
    const State& world_leaf_state(int world) const{return worlds.at(world).leaf.state;}
    const std::vector<Action>& world_leaf_actions(int world) const{return worlds.at(world).leaf.actions;}
    int pending_rows(bool state_only) const {
        int count=0;
        if(!state_only && root_exportable())++count;
        for(auto& w:worlds)if(w.has_leaf && !w.leaf.exported && w.leaf.cap==state_only)++count;
        return count;
    }
    // Policy rows report their widest candidate list; value-only rows carry no
    // candidates, so they only report whether any row is waiting.
    int pending_width(bool state_only) const {
        if(state_only)return pending_rows(true)>0?1:0;
        int best=0;
        if(root_exportable())best=int(root_actions.size());
        for(auto& w:worlds)if(w.has_leaf && !w.leaf.exported && !w.leaf.cap)best=std::max(best,int(w.leaf.actions.size()));
        return best;
    }
    // Rejects stale or duplicated rows for one exported (world, simulation) tag
    // without touching the tree.  A batch verifies every row before any of them
    // is applied, so a bad row cannot leave a half-applied batch behind.
    void verify_exported(int world,int simulation,bool has_logits,int width) const {
        int count=0;
        if(world<0){
            if(!root_pending_now())throw std::logic_error("root request already resolved");
            if(!has_logits)throw std::invalid_argument("root request needs logits");
            count=int(root_actions.size());
        } else {
            if(!world_has_leaf(world))throw std::logic_error("world has no pending leaf");
            if(world_leaf_simulation(world)!=simulation)throw std::logic_error("stale batch row for this world");
            if(!worlds.at(world).leaf.exported)throw std::logic_error("batch row was not exported by this buffer");
            bool cap=world_leaf_cap(world);
            if(cap && has_logits)throw std::invalid_argument("value-only leaf cannot receive logits");
            if(!cap && !has_logits)throw std::invalid_argument("policy leaf needs logits");
            count=cap?0:int(world_leaf_actions(world).size());
        }
        if(has_logits && width<count)throw std::invalid_argument("batch row is narrower than the leaf action list");
    }
    // Consumes the exported row that ``verify_exported`` accepted.
    void submit_exported(int world,int simulation,const float* logits,int width,double value) {
        verify_exported(world,simulation,logits!=nullptr,width);
        if(world<0)root_exported=false;
        else worlds.at(world).leaf.exported=false;
        int count=0;
        if(logits!=nullptr){
            count = world<0 ? int(root_actions.size())
                            : (world_leaf_cap(world)?0:int(world_leaf_actions(world).size()));
        }
        std::vector<double> copied;
        if(logits!=nullptr){copied.assign(logits,logits+count);}
        submit(world,copied,value);
    }
    // Root worlds need an observation only at construction. Instantiate before returning
    // the first root evaluation; no Python object is retained in the search tree.
    void initialize_worlds(py::object request) {
        if(root_pending || !worlds.empty() || finished) return;
        worlds.resize(cfg.determinizations);
        py::object obs=request.attr("observation");
        for(int j=0;j<cfg.determinizations;++j) {
            World& w=worlds[j];w.budget=budgets[j];w.nodes.reserve(w.budget+1);
            State s=sampled_state(obs,catalog,derived_seed(public_word,base_seed,j));
            auto root=decide(s,catalog);
            if(root!=root_actions)throw std::invalid_argument("sampled world root candidates differ");
            w.nodes.emplace_back(std::move(s),root,used_prior);
            if(stochastic)w.information_nodes.emplace(information_signature(root_state,catalog,root_state.actor),0);
        }
        py::gil_scoped_release release;
        for(int j=0;j<cfg.determinizations;++j)advance(j);
    }
    py::dict failure_context(int requested_world) const {
        int world=requested_world;
        if(world<0 || world>=int(worlds.size()) || !worlds[world].has_leaf)world=current_world;
        py::dict out;out["world_index"]=world>=0?py::cast(world):py::none();
        out["simulation_index"]=world>=0?py::cast(worlds[world].has_leaf?worlds[world].leaf.simulation:current_simulation):py::none();
        if(stochastic) {
            int simulation=world>=0?(worlds[world].has_leaf?worlds[world].leaf.simulation:current_simulation):0;
            out["hidden_sample_seed"]=derived_seed(public_word,base_seed,simulation);
            out["chance_seed"]=derived_seed(public_word,base_seed^0x4348414E4345ULL,simulation);
        }
        out["determinization_seed"]=world>=0?py::cast(derived_seed(public_word,base_seed,world)):py::none();
        auto& branch=(world>=0 && worlds[world].has_leaf)?worlds[world].leaf.branch:current_branch;
        py::list items;
        for(auto& a:branch)items.append(py::make_tuple(a.type,a.id,a.tier,a.tokens));
        out["branch_actions"]=items;
        if(has_attempted)out["attempted_action"]=py::make_tuple(current_attempted.type,current_attempted.id,current_attempted.tier,current_attempted.tokens);
        else out["attempted_action"]=py::none();
        return out;
    }
    py::dict result() const {
        if(!finished)throw std::logic_error("search unfinished");
        py::dict out;out["chosen_candidate_index"]=chosen;
        out["searchable"]=!worlds.empty();out["root_network_prior"]=network_prior;
        out["root_used_prior"]=used_prior;out["simulation_budgets"]=budgets;
        out["per_world_root_visit_counts"]=rows;out["root_visit_counts"]=counts;
        out["policy_target"]=target;out["behavior_policy"]=behavior;
        out["simulations"]=int(worlds.empty()?0:cfg.simulations);
        out["max_tree_depth"]=max_depth;out["transition_cap_hits"]=cap_hits;
        return out;
    }
    py::dict stats() const {
        py::dict out;out["simulations"]=int(std::accumulate(worlds.begin(),worlds.end(),0,[](int a,const World& w){return a+w.completed;}));
        out["determinizations"]=int(worlds.size());out["expanded_nodes"]=expanded;
        out["state_transitions"]=transitions;out["folded_actions"]=folded;
        out["terminal_hits"]=terminal_hits;out["transition_cap_hits"]=cap_hits;
        out["tree_depth_sum"]=tree_depth_sum;out["model_positions"]=model_positions;
        out["rule_seconds"]=rule_seconds;out["feature_seconds"]=feature_seconds;
        out["native_compute_seconds"]=compute_seconds;
        return out;
    }
    py::dict chance_diagnostics() const {
        py::dict out;out["stochastic"]=stochastic;out["root_hidden_samples"]=root_samples;
        out["chance_draws"]=chance_draws;out["information_set_hits"]=information_hits;
        out["decision_nodes"]=worlds.empty()?0:int(worlds[0].nodes.size());
        py::list outcomes;std::vector<double> values;
        if(!worlds.empty())for(auto& edge:worlds[0].nodes[0].edges) {
            std::vector<int> counts;if(edge.outcomes)for(auto& item:*edge.outcomes)counts.push_back(item.second);
            std::sort(counts.begin(),counts.end());outcomes.append(counts);values.push_back(edge.value_sum);
        }
        out["root_outcome_visits"]=outcomes;out["root_value_sums"]=values;
        if(chance_pw_k>0) {
            out["chance_widening"]=py::dict("k"_a=chance_pw_k,"alpha"_a=chance_pw_alpha);
            out["chance_pw_fresh_samples"]=chance_pw_fresh;out["chance_pw_reuses"]=chance_pw_reused;
            int contexts=0,support=0;py::list root_pools;
            if(!worlds.empty())for(size_t n=0;n<worlds[0].nodes.size();++n) {
                for(auto& edge:worlds[0].nodes[n].edges) {
                    py::list pools;
                    if(edge.draw_pools)for(auto& item:*edge.draw_pools) {
                        auto& pool=item.second;++contexts;support+=int(pool.outcomes.size());
                        if(n==0)pools.append(py::dict("visits"_a=pool.visits,"pool_size"_a=pool.pool_size,
                                                     "outcomes"_a=pool.outcomes));
                    }
                    if(n==0)root_pools.append(pools);
                }
            }
            out["chance_pw_contexts"]=contexts;out["chance_pw_admitted_outcomes"]=support;
            out["root_chance_pools"]=root_pools;
        }
        return out;
    }
    py::dict debug_leaf_state(int world) const {
        if(!worlds.at(world).has_leaf)throw std::logic_error("no pending search leaf");
        return state_snapshot(worlds.at(world).leaf.state);
    }
};
// One contiguous batch of encoded pending leaves.  Every float field lives in
// one packed blob and every mask in a second packed uint8 blob, so a native fill
// writes the final transport layout directly and a device copy only needs the
// used prefix.  Storage is either allocated here or borrowed from Python
// (pinned staging tensors on the CUDA path); the returned views keep the owner
// alive, and ``submit``/``submit_fused`` consume exactly the filled rows.
class NativeBatch {
    const Catalog& catalog;
    bool state_only;
    int capacity;
    int width;
    int rows=0;
    int float_stride=0, mask_stride=0;
    int global_off=0, self_off=0, opponent_off=0, noble_off=0, pending_off=0, candidate_off=0;
    std::array<int,3> market_off{};
    int market_mask_off_[3]={0,0,0};
    int self_mask_off=0, opponent_mask_off=0, noble_mask_off=0, pending_mask_off=0, candidate_mask_off=0;
    std::vector<float> owned_float;
    std::vector<uint8_t> owned_mask;
    py::object float_owner, mask_owner;  // borrowed staging arrays; none -> owned blobs
    float* float_base=nullptr;
    uint8_t* mask_base=nullptr;
    std::vector<std::shared_ptr<SearchJob>> owners;
    std::vector<int> worlds, simulations, kinds, counts;

    // Views borrow the packed blobs; the passed owner keeps them alive.
    static py::array_t<float> float_view(const float* data,int rows_,int cols,int depth,py::ssize_t row_stride,py::handle base){
        if(depth<0)return py::array_t<float>({(py::ssize_t)rows_,(py::ssize_t)cols},
                                             {(py::ssize_t)(row_stride*py::ssize_t(sizeof(float))),(py::ssize_t)sizeof(float)},data,base);
        return py::array_t<float>({(py::ssize_t)rows_,(py::ssize_t)cols,(py::ssize_t)depth},
                                  {(py::ssize_t)(row_stride*py::ssize_t(sizeof(float))),(py::ssize_t)(depth*py::ssize_t(sizeof(float))),(py::ssize_t)sizeof(float)},data,base);
    }
    static py::array_t<float> float_view_1d(const float* data,py::ssize_t count,py::handle base){
        return py::array_t<float>({count},{(py::ssize_t)sizeof(float)},data,base);
    }
    static py::array_t<bool> mask_view(const uint8_t* data,int rows_,int cols,py::ssize_t row_stride,py::handle base){
        if(cols<0)return py::array_t<bool>({(py::ssize_t)rows_},{(py::ssize_t)row_stride},reinterpret_cast<const bool*>(data),base);
        return py::array_t<bool>({(py::ssize_t)rows_,(py::ssize_t)cols},
                                 {(py::ssize_t)row_stride,(py::ssize_t)sizeof(bool)},reinterpret_cast<const bool*>(data),base);
    }
    static py::array_t<uint8_t> mask_view_1d(const uint8_t* data,py::ssize_t count,py::handle base){
        return py::array_t<uint8_t>({count},{(py::ssize_t)sizeof(uint8_t)},data,base);
    }
    static py::tuple field(const std::string& name,int offset,int count,std::initializer_list<int> shape){
        py::tuple dims(shape.size());size_t index=0;
        for(int value:shape)dims[index++]=value;
        return py::make_tuple(name,offset,count,dims);
    }
    static void require_staging(py::array& raw,size_t count,const char* label,bool mask){
        std::string prefix=std::string(label)+" staging must be ";
        if(raw.ndim()!=1)throw std::invalid_argument(prefix+"one dimensional");
        if(!(raw.flags()&py::array::c_style))throw std::invalid_argument(prefix+"contiguous");
        if(size_t(raw.shape(0))<count)throw std::invalid_argument(prefix+"large enough for the batch capacity");
        if(!py::cast<bool>(raw.attr("flags").attr("writeable")))throw std::invalid_argument(prefix+"writable");
        bool dtype_ok = mask ? (raw.dtype().is(py::dtype::of<uint8_t>())||raw.dtype().is(py::dtype::of<bool>()))
                             : raw.dtype().is(py::dtype::of<float>());
        if(!dtype_ok)throw std::invalid_argument(prefix+std::string(mask?"uint8 or bool":"float32"));
    }
    void write_cards(float* features,uint8_t* mask,const std::vector<CardRow>& rows_,int slot_capacity,const State& s,int observer){
        std::fill(features,features+size_t(slot_capacity)*CARD_DIM,0.0f);
        std::fill(mask,mask+slot_capacity,uint8_t(0));
        for(size_t i=0;i<rows_.size();++i){
            auto f=card_features(rows_[i],s,catalog,observer);
            std::copy(f.begin(),f.end(),features+size_t(i)*CARD_DIM);
            mask[i]=1;
        }
    }
    void fill_row(int row,const State& s,const std::vector<Action>& acts){
        int observer=s.actor;
        float* features=float_base+size_t(row)*float_stride;
        uint8_t* mask=mask_base+size_t(row)*mask_stride;
        auto g=global_features(s,catalog,observer);
        std::copy(g.begin(),g.end(),features+global_off);
        for(int t=0;t<3;++t){
            std::vector<int> ids=s.market[t];
            std::sort(ids.begin(),ids.end(),[&](int a,int b){return card_less(a,b,catalog);});
            std::vector<CardRow> card_rows;for(int id:ids)card_rows.push_back({id,t,0});
            write_cards(features+market_off[t],mask+market_mask_off_[t],card_rows,4,s,observer);
        }
        auto own=reservation_rows(s,catalog,observer,observer),opp=reservation_rows(s,catalog,1-observer,observer);
        write_cards(features+self_off,mask+self_mask_off,own,3,s,observer);
        write_cards(features+opponent_off,mask+opponent_mask_off,opp,3,s,observer);
        auto nob=sorted_nobles(s.nobles,catalog);
        if(nob.size()>3)throw std::invalid_argument("native noble row exceeds feature capacity");
        float* noble_block=features+noble_off;
        uint8_t* noble_mask=mask+noble_mask_off;
        std::fill(noble_block,noble_block+3*NOBLE_DIM,0.0f);std::fill(noble_mask,noble_mask+3,uint8_t(0));
        for(size_t i=0;i<nob.size();++i){
            auto f=noble_features(nob[i],s,catalog,observer);
            std::copy(f.begin(),f.end(),noble_block+i*NOBLE_DIM);noble_mask[i]=1;
        }
        bool pending=s.phase==1;
        CardRow pending_row;
        if(pending){Action payment;payment.type=6;pending_row=target_card(s,catalog,observer,payment);}
        auto pf=card_features(pending_row,s,catalog,observer);
        std::copy(pf.begin(),pf.end(),features+pending_off);
        mask[pending_mask_off]=pending?1:0;
        if(state_only)return;
        float* dst=features+candidate_off;
        uint8_t* candidate=candidate_mask_off+mask;
        std::fill(dst,dst+size_t(width)*ACTION_DIM,0.0f);std::fill(candidate,candidate+width,uint8_t(0));
        if(acts.size()>size_t(width))throw std::invalid_argument("candidate count exceeds the batch width bucket");
        for(size_t i=0;i<acts.size();++i){
            auto f=action_features(s,catalog,observer,acts[i]);
            std::copy(f.begin(),f.end(),dst+i*ACTION_DIM);candidate[i]=1;
        }
    }
    static std::string row_message(int row,const std::string& detail){
        return "batch row "+std::to_string(row)+": "+detail;
    }
    void check_row(int row,const float* logits,double value) const {
        if(!std::isfinite(value))throw std::invalid_argument(row_message(row,"nonfinite network value"));
        if(logits!=nullptr){
            for(int i=0;i<counts[row];++i)
                if(!std::isfinite(logits[i]))throw std::invalid_argument(row_message(row,"nonfinite network logits"));
        }
        try{ owners[row]->verify_exported(worlds[row],simulations[row],logits!=nullptr,width); }
        catch(const std::exception& error){ throw std::invalid_argument(row_message(row,error.what())); }
    }
    template <typename LogitsAt,typename ValueAt>
    void apply_batch(LogitsAt logits_at,ValueAt value_at,bool has_logits){
        for(int row=0;row<rows;++row)check_row(row,has_logits?logits_at(row):nullptr,value_at(row));
        for(int row=0;row<rows;++row)
            owners[row]->submit_exported(worlds[row],simulations[row],has_logits?logits_at(row):nullptr,width,value_at(row));
        reset();
    }
public:
    NativeBatch(const Catalog& catalog_,bool state_only_,int capacity_,int width_,
                py::object float_staging=py::none(),py::object mask_staging=py::none())
        :catalog(catalog_),state_only(state_only_),capacity(capacity_),width(width_) {
        if(capacity<=0)throw std::invalid_argument("batch capacity must be positive");
        if(width<=0)throw std::invalid_argument("batch width must be positive");
        int cursor=0;
        global_off=cursor;cursor+=GLOBAL_DIM;
        for(int t=0;t<3;++t){market_off[t]=cursor;cursor+=4*CARD_DIM;}
        self_off=cursor;cursor+=3*CARD_DIM;
        opponent_off=cursor;cursor+=3*CARD_DIM;
        noble_off=cursor;cursor+=3*NOBLE_DIM;
        pending_off=cursor;cursor+=CARD_DIM;
        if(!state_only){candidate_off=cursor;cursor+=width*ACTION_DIM;}
        float_stride=cursor;
        cursor=0;
        for(int t=0;t<3;++t){market_mask_off_[t]=cursor;cursor+=4;}
        self_mask_off=cursor;cursor+=3;
        opponent_mask_off=cursor;cursor+=3;
        noble_mask_off=cursor;cursor+=3;
        pending_mask_off=cursor;cursor+=1;
        if(!state_only){candidate_mask_off=cursor;cursor+=width;}
        mask_stride=cursor;
        if(float_staging.is_none()){
            owned_float.assign(size_t(capacity)*float_stride,0.0f);
            float_base=owned_float.data();
        } else {
            py::array raw=py::reinterpret_borrow<py::array>(float_staging);
            require_staging(raw,size_t(capacity)*float_stride,"float",false);
            float_owner=float_staging;float_base=static_cast<float*>(raw.mutable_data());
        }
        if(mask_staging.is_none()){
            owned_mask.assign(size_t(capacity)*mask_stride,uint8_t(0));
            mask_base=owned_mask.data();
        } else {
            py::array raw=py::reinterpret_borrow<py::array>(mask_staging);
            require_staging(raw,size_t(capacity)*mask_stride,"mask",true);
            mask_owner=mask_staging;mask_base=static_cast<uint8_t*>(raw.mutable_data());
        }
        owners.reserve(capacity);worlds.reserve(capacity);
    }
    bool is_state_only() const{return state_only;}
    int row_count() const{return rows;}
    int padded_width() const{return state_only?0:width;}
    int row_capacity() const{return capacity;}
    int float_stride_elements() const{return float_stride;}
    int mask_stride_elements() const{return mask_stride;}
    void reset(){rows=0;owners.clear();worlds.clear();simulations.clear();kinds.clear();counts.clear();}
    // Each pending leaf is claimed at most once between fills and submits.
    int fill(std::shared_ptr<SearchJob> job) {
        int start=rows;
        if(rows<capacity && !state_only && job->root_exportable()){
            job->mark_root_exported();
            append(job,-1,-1,0,int(job->root_actions_ref().size()),job->root_state_ref(),job->root_actions_ref());
        }
        for(int world=0;world<job->world_count() && rows<capacity;++world){
            if(!job->world_leaf_exportable(world))continue;
            if(job->world_leaf_cap(world)!=state_only)continue;
            int simulation=job->world_leaf_simulation(world);
            bool cap=job->world_leaf_cap(world);
            job->mark_leaf_exported(world);
            append(job,world,simulation,cap?2:1,cap?0:int(job->world_leaf_actions(world).size()),
                   job->world_leaf_state(world),job->world_leaf_actions(world));
        }
        return rows-start;
    }
    void append(std::shared_ptr<SearchJob> job,int world,int simulation,int kind,int count,const State& s,const std::vector<Action>& acts) {
        if(rows>=capacity)throw std::logic_error("batch buffer is full");
        int row=rows++;
        owners.push_back(job);worlds.push_back(world);simulations.push_back(simulation);
        kinds.push_back(kind);counts.push_back(count);
        fill_row(row,s,acts);
    }
    // (world, simulation, kind, valid candidate count) per filled row.
    py::list row_ids() const {
        py::list result;
        for(int row=0;row<rows;++row)
            result.append(py::make_tuple(worlds[row],simulations[row],kinds[row],counts[row]));
        return result;
    }
    // Same identity without per-row Python objects; the scheduling hot path
    // uses this and expands readable rows only when a failure has to be logged.
    py::array_t<int64_t> row_ids_array() const {
        py::array_t<int64_t> out({(py::ssize_t)rows,(py::ssize_t)4});
        int64_t* data=out.mutable_data();
        for(int row=0;row<rows;++row){
            data[row*4+0]=worlds[row];data[row*4+1]=simulations[row];
            data[row*4+2]=kinds[row];data[row*4+3]=counts[row];
        }
        return out;
    }
    py::dict arrays(py::handle base) const {
        py::object fbase=float_owner.is_none()?py::reinterpret_borrow<py::object>(base):float_owner;
        py::object mbase=mask_owner.is_none()?py::reinterpret_borrow<py::object>(base):mask_owner;
        py::dict out;
        out["feature_schema_version"]="splendor-observation-action-v1";
        out["global_features"]=float_view(float_base+global_off,rows,GLOBAL_DIM,-1,float_stride,fbase);
        for(int t=0;t<3;++t){
            std::string key="market_t"+std::to_string(t+1);
            out[py::str(key)]=float_view(float_base+market_off[t],rows,4,CARD_DIM,float_stride,fbase);
            out[py::str(key+"_mask")]=mask_view(mask_base+market_mask_off_[t],rows,4,mask_stride,mbase);
        }
        out["self_reserved"]=float_view(float_base+self_off,rows,3,CARD_DIM,float_stride,fbase);
        out["self_reserved_mask"]=mask_view(mask_base+self_mask_off,rows,3,mask_stride,mbase);
        out["opponent_reserved"]=float_view(float_base+opponent_off,rows,3,CARD_DIM,float_stride,fbase);
        out["opponent_reserved_mask"]=mask_view(mask_base+opponent_mask_off,rows,3,mask_stride,mbase);
        out["nobles"]=float_view(float_base+noble_off,rows,3,NOBLE_DIM,float_stride,fbase);
        out["nobles_mask"]=mask_view(mask_base+noble_mask_off,rows,3,mask_stride,mbase);
        out["pending_card"]=float_view(float_base+pending_off,rows,CARD_DIM,-1,float_stride,fbase);
        out["pending_card_mask"]=mask_view(mask_base+pending_mask_off,rows,-1,mask_stride,mbase);
        if(!state_only){
            out["candidate_features"]=float_view(float_base+candidate_off,rows,width,ACTION_DIM,float_stride,fbase);
            out["candidate_mask"]=mask_view(mask_base+candidate_mask_off,rows,width,mask_stride,mbase);
        }
        return out;
    }
    py::dict packed(py::handle base) const {
        py::object fbase=float_owner.is_none()?py::reinterpret_borrow<py::object>(base):float_owner;
        py::object mbase=mask_owner.is_none()?py::reinterpret_borrow<py::object>(base):mask_owner;
        py::dict out;
        out["rows"]=rows;out["width"]=state_only?0:width;out["state_only"]=state_only;
        out["capacity"]=capacity;out["float_stride"]=float_stride;out["mask_stride"]=mask_stride;
        out["float_blob"]=float_view_1d(float_base,(py::ssize_t)capacity*float_stride,fbase);
        out["mask_blob"]=mask_view_1d(mask_base,(py::ssize_t)capacity*mask_stride,mbase);
        py::list float_fields,mask_fields;
        float_fields.append(field("global_features",global_off,GLOBAL_DIM,{GLOBAL_DIM}));
        for(int t=0;t<3;++t)
            float_fields.append(field("market_t"+std::to_string(t+1),market_off[t],4*CARD_DIM,{4,CARD_DIM}));
        float_fields.append(field("self_reserved",self_off,3*CARD_DIM,{3,CARD_DIM}));
        float_fields.append(field("opponent_reserved",opponent_off,3*CARD_DIM,{3,CARD_DIM}));
        float_fields.append(field("nobles",noble_off,3*NOBLE_DIM,{3,NOBLE_DIM}));
        float_fields.append(field("pending_card",pending_off,CARD_DIM,{CARD_DIM}));
        for(int t=0;t<3;++t)
            mask_fields.append(field("market_t"+std::to_string(t+1)+"_mask",market_mask_off_[t],4,{4}));
        mask_fields.append(field("self_reserved_mask",self_mask_off,3,{3}));
        mask_fields.append(field("opponent_reserved_mask",opponent_mask_off,3,{3}));
        mask_fields.append(field("nobles_mask",noble_mask_off,3,{3}));
        mask_fields.append(field("pending_card_mask",pending_mask_off,1,{}));
        if(!state_only){
            float_fields.append(field("candidate_features",candidate_off,width*ACTION_DIM,{width,ACTION_DIM}));
            mask_fields.append(field("candidate_mask",candidate_mask_off,width,{width}));
        }
        out["float_fields"]=float_fields;out["mask_fields"]=mask_fields;
        return out;
    }
    void submit(py::object logits,py::object values) {
        // The converted arrays outlive the whole submission: a forcecast of a
        // float64 or non-contiguous input owns temporary storage.
        auto value_array=py::array_t<float,py::array::c_style|py::array::forcecast>(values);
        if(value_array.ndim()!=1 || value_array.shape(0)!=rows)
            throw std::invalid_argument("batch values must have one entry per row");
        py::array_t<float,py::array::c_style|py::array::forcecast> logit_array;
        const float* logits_ptr=nullptr;
        if(!state_only){
            logit_array=py::array_t<float,py::array::c_style|py::array::forcecast>(logits);
            if(logit_array.ndim()!=2 || logit_array.shape(0)!=rows || logit_array.shape(1)!=width)
                throw std::invalid_argument("batch logits must have shape (rows, padded width)");
            logits_ptr=logit_array.data();
        } else if(!logits.is_none()){
            throw std::invalid_argument("value-only batches take no logits");
        }
        const float* values_ptr=value_array.data();
        apply_batch([&](int row){return logits_ptr==nullptr?nullptr:logits_ptr+size_t(row)*width;},
                    [&](int row){return double(values_ptr[row]);},logits_ptr!=nullptr);
    }
    void submit_fused(py::object fused) {
        if(state_only)throw std::invalid_argument("value-only batches take no fused logits");
        auto array=py::array_t<float,py::array::c_style|py::array::forcecast>(fused);
        if(array.ndim()!=2 || array.shape(0)!=rows || array.shape(1)!=width+1)
            throw std::invalid_argument("fused batch must have shape (rows, padded width + 1)");
        const float* data=array.data();
        apply_batch([&](int row){return data+size_t(row)*(width+1);},
                    [&](int row){return double(data[size_t(row)*(width+1)+width]);},true);
    }
};
static State truth_state(py::object source,const Catalog& c) {
    State s;s.supply=token_vector(source.attr("supply"));
    s.first=py::cast<int>(source.attr("first_player"));s.actor=py::cast<int>(source.attr("current_player"));
    auto turns=py::reinterpret_borrow<py::tuple>(source.attr("completed_turns"));
    s.turns[0]=py::cast<int>(turns[0]);s.turns[1]=py::cast<int>(turns[1]);
    s.skips=py::cast<int>(source.attr("consecutive_forced_skips"));
    s.phase=phase_id(enum_value(source.attr("phase")));
    if(!source.attr("final_round_triggered_by").is_none())s.trigger=py::cast<int>(source.attr("final_round_triggered_by"));
    if(s.phase==4) {
        py::object winners=source.attr("result").attr("winner_ids");
        s.winner=py::len(winners)==2?-1:py::cast<int>(winners[0]);
    }
    for(int t=0;t<3;++t) {
        const char* name=t==0?"tier_one":t==1?"tier_two":"tier_three";
        for(py::handle id:source.attr("decks").attr(name))s.decks[t].push_back(c.card(id));
        for(py::handle id:source.attr("market").attr(name))s.market[t].push_back(c.card(id));
    }
    for(int i=0;i<2;++i) {
        py::handle p=py::reinterpret_borrow<py::tuple>(source.attr("players"))[i];
        s.players[i].tokens=token_vector(p.attr("tokens"));
        s.players[i].purchased=purchased_ids(p,c);s.players[i].nobles=noble_ids(p,c);
        for(py::handle r:p.attr("reserved_cards"))s.players[i].reserved.push_back({c.card(r.attr("card_id")),enum_value(r.attr("source"))=="market"?0:1});
    }
    for(py::handle id:source.attr("nobles_in_play"))s.nobles.push_back(c.noble(id));
    if(!source.attr("pending").is_none()) {
        py::object pending=source.attr("pending");s.has_pending=true;
        s.pending=parse_action(pending.attr("main_action"),c);
        s.return_count=py::cast<int>(pending.attr("tokens_to_return"));
        for(py::handle id:pending.attr("eligible_noble_ids"))s.eligible.push_back(c.noble(id));
    }
    return s;
}
static py::tuple code_action(const Action& a) {return py::make_tuple(a.type,a.id,a.tier,a.tokens);}
static py::dict state_snapshot(const State& state) {
    py::dict out;out["phase"]=state.phase;out["actor"]=state.actor;
    out["first"]=state.first;out["supply"]=state.supply;out["turns"]=py::make_tuple(state.turns[0],state.turns[1]);
    out["skips"]=state.skips;out["forced_skip_total"]=state.forced_skip_total;out["trigger"]=state.trigger;out["winner"]=state.winner;
    out["decks"]=state.decks;out["market"]=state.market;out["nobles"]=state.nobles;
    py::list players;
    for(auto& p:state.players) {
        py::dict item;item["tokens"]=p.tokens;item["purchased"]=p.purchased;item["nobles"]=p.nobles;
        py::list reserved;for(auto& r:p.reserved)reserved.append(py::make_tuple(r.card,r.source));
        item["reserved"]=reserved;players.append(item);
    }
    out["players"]=players;
    out["pending"]=state.has_pending?py::object(code_action(state.pending)):py::none();
    out["return_count"]=state.return_count;out["eligible"]=state.eligible;return out;
}
class NativeGame {
    const Catalog& catalog;State state;std::vector<Action> legal;
public:
    NativeGame(const Catalog& c,py::object source):catalog(c),state(truth_state(source,c)){legal=decide(state,catalog);}
    py::list actions() const {py::list result;for(auto& a:legal)result.append(code_action(a));return result;}
    bool verify_actions(py::object request) const {
        std::vector<Action> reference;
        for(py::handle a:request.attr("legal_actions"))reference.push_back(parse_action(a,catalog));
        return reference==legal;
    }
    void apply_index(int index) {
        if(index<0 || index>=int(legal.size()))throw std::out_of_range("candidate index outside native legal actions");
        apply(state,legal[index],catalog);legal=decide(state,catalog);
    }
    py::dict snapshot() const {
        return state_snapshot(state);
    }
    py::dict encoded() const {if(state.phase==4)throw std::logic_error("terminal state has no decision features");return encode(state,legal,catalog);}
};
PYBIND11_MODULE(_native_search,m) {
    m.def("information_key",[](const Catalog& c,py::object source,int observer) {
        if(observer<0 || observer>1)throw std::invalid_argument("observer must be 0 or 1");
        return py::bytes(information_signature(truth_state(source,c),c,observer));
    });
    py::class_<Catalog>(m,"Catalog")
        .def(py::init<py::object>());
    py::class_<NativeGame>(m,"NativeGame")
        .def(py::init<const Catalog&,py::object>(),py::keep_alive<1,2>())
        .def("actions",&NativeGame::actions)
        .def("verify_actions",&NativeGame::verify_actions)
        .def("apply_index",&NativeGame::apply_index)
        .def("snapshot",&NativeGame::snapshot)
        .def("encoded",&NativeGame::encoded);
    py::class_<NativeBatch,std::shared_ptr<NativeBatch>>(m,"NativeBatch")
        .def(py::init<const Catalog&,bool,int,int,py::object,py::object>(),
             py::arg("catalog"),py::arg("state_only"),py::arg("capacity"),py::arg("width"),
             py::arg("float_staging")=py::none(),py::arg("mask_staging")=py::none(),
             py::keep_alive<1,2>())
        .def("is_state_only",&NativeBatch::is_state_only)
        .def("row_count",&NativeBatch::row_count)
        .def("row_ids",&NativeBatch::row_ids)
        .def("row_ids_array",&NativeBatch::row_ids_array)
        .def("padded_width",&NativeBatch::padded_width)
        .def("row_capacity",&NativeBatch::row_capacity)
        .def("float_stride",&NativeBatch::float_stride_elements)
        .def("mask_stride",&NativeBatch::mask_stride_elements)
        .def("reset",&NativeBatch::reset)
        .def("fill",&NativeBatch::fill)
        .def("arrays",[](const std::shared_ptr<NativeBatch>& self){return self->arrays(py::cast(self));})
        .def("packed",[](const std::shared_ptr<NativeBatch>& self){return self->packed(py::cast(self));})
        .def("submit",&NativeBatch::submit)
        .def("submit_fused",&NativeBatch::submit_fused);
    py::class_<SearchJob,std::shared_ptr<SearchJob>>(m,"SearchJob")
        .def(py::init<const Catalog&,py::object,py::object,uint64_t,uint64_t,bool,double,bool,double,double>(),
             py::arg("catalog"),py::arg("request"),py::arg("config"),py::arg("public_word"),
             py::arg("sampling_seed"),py::arg("add_root_noise"),py::arg("temperature"),py::arg("stochastic")=false,
             py::arg("chance_pw_k")=0.,py::arg("chance_pw_alpha")=.5,
             py::keep_alive<1,2>())
        .def("done",&SearchJob::done)
        .def("pending",&SearchJob::pending)
        .def("pending_count",&SearchJob::pending_count)
        .def("submit",&SearchJob::submit)
        .def("initialize_worlds",&SearchJob::initialize_worlds)
        .def("result",&SearchJob::result)
        .def("stats",&SearchJob::stats)
        .def("chance_diagnostics",&SearchJob::chance_diagnostics)
        .def("debug_leaf_state",&SearchJob::debug_leaf_state)
        .def("failure_context",&SearchJob::failure_context)
        .def("pending_width",&SearchJob::pending_width)
        .def("pending_rows",&SearchJob::pending_rows)
        .def("root_pending_now",&SearchJob::root_pending_now);
}
