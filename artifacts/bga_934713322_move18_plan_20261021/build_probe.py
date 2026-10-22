"""Build an experiment-only native copy with observational node export."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import sysconfig
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BUILD = ROOT / 'runtime/bga_934713322_move18_plan_20261021/probe'


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    original = (ROOT / 'src/splendor/native_search.cpp').read_text()
    text = original.replace('std::vector<Node> nodes;',
        'std::vector<Node> nodes;\n    std::vector<std::vector<Action>> debug_branches;')
    needle = 'w.nodes.emplace_back(std::move(s),root,used_prior);'
    assert text.count(needle) == 1
    text = text.replace(needle, needle + '\n            w.debug_branches.emplace_back();')
    needle = 'int child=int(w.nodes.size());w.nodes.emplace_back(std::move(leaf.state),std::move(leaf.actions),std::move(prior));'
    assert text.count(needle) == 1
    text = text.replace(needle, needle + '\n            w.debug_branches.push_back(leaf.branch);')
    method = r'''
    py::list debug_early_nodes() const {
        if(!finished)throw std::logic_error("export only after search finishes");
        py::list rows;
        for(size_t world=0;world<worlds.size();++world) {
            const auto& w=worlds[world];
            for(size_t node=0;node<w.nodes.size();++node) {
                const auto& n=w.nodes[node];const auto& s=n.state;
                if(s.phase!=0 || s.turns[0]>root_state.turns[0]+1 || s.turns[1]>root_state.turns[1]+1)continue;
                py::dict row;row["world"]=world;row["node"]=node;
                row["state"]=state_snapshot(s);row["prior"]=n.prior;
                py::list branch,actions;std::vector<int> visits,children;std::vector<double> values;
                for(const auto& a:w.debug_branches.at(node))branch.append(py::make_tuple(a.type,a.id,a.tier,a.tokens));
                for(size_t i=0;i<n.actions.size();++i) {
                    const auto& a=n.actions[i];const auto& e=n.edges[i];
                    actions.append(py::make_tuple(a.type,a.id,a.tier,a.tokens));
                    visits.push_back(e.visits);values.push_back(e.value_sum);children.push_back(e.child);
                }
                row["branch"]=branch;row["actions"]=actions;row["visits"]=visits;
                row["value_sums_root_actor"]=values;row["children"]=children;rows.append(row);
            }
        }
        return rows;
    }
'''
    needle = '    py::dict debug_leaf_state(int world) const {'
    assert text.count(needle) == 1
    text = text.replace(needle, method + '\n' + needle)
    needle = '.def("debug_leaf_state",&SearchJob::debug_leaf_state)'
    assert text.count(needle) == 1
    text = text.replace(needle, '.def("debug_early_nodes",&SearchJob::debug_early_nodes)\n        ' + needle)
    source = BUILD / 'native_search_probe.cpp'
    source.write_text(text)
    binary = BUILD / ('_native_search' + sysconfig.get_config_var('EXT_SUFFIX'))
    # Torch provides the installed pybind11 headers; no environment install needed.
    headers = Path(torch.__file__).parent / 'include'
    assert (headers / 'pybind11/pybind11.h').is_file()
    command = ['c++', '-O3', '-DNDEBUG', '-std=c++17', '-shared', '-fPIC',
               '-fvisibility=hidden', '-I' + str(headers),
               '-I' + sysconfig.get_path('include'), str(source), '-o', str(binary)]
    subprocess.run(command, cwd=BUILD, check=True)
    # Save the exact diagnostic delta, not another long-lived production engine.
    import difflib
    (OUT / 'native_probe.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), text.splitlines(True), fromfile='src/splendor/native_search.cpp',
        tofile='runtime/probe/native_search_probe.cpp')))
    (OUT / 'probe_build.json').write_text(json.dumps({
        'production_source_sha256': hashlib.sha256(original.encode()).hexdigest(),
        'probe_source_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'binary': str(binary.relative_to(ROOT)),
        'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'python': sys.version, 'command': command,
        'scope': 'Record first-arrival branches and export node statistics; search selection and backup unchanged.'
    }, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
