import networkx as nx
from collections import deque
import walker
from collections import defaultdict
def build_graph(graph: list) -> nx.Graph:
    G = nx.Graph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h, t, relation=r.strip())
    return G

def build_digraph(graph: list) -> nx.DiGraph:
    G = nx.DiGraph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h, t, relation=r.strip())
    return G

def get_truth_paths(q_entity: list, a_entity: list, graph: nx.Graph) -> list:
    '''
    Get shortest paths connecting question and answer entities.
    '''
    paths = []
    for h in q_entity:
        if h not in graph:
            continue
        for t in a_entity:
            if t not in graph:
                continue
            try:
                for p in nx.all_shortest_paths(graph, h, t):
                    paths.append(p)
            except:
                pass
    # Add relation to paths
    result_paths = []
    for p in paths:
        tmp = []
        for i in range(len(p)-1):
            u = p[i]
            v = p[i+1]
            tmp.append((u, graph[u][v]['relation'], v))
        result_paths.append(tmp)
    return result_paths

def build_graph_new(triples) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for h, r, t in triples:
        r = str(r).strip()
        G.add_edge(str(h), str(t), key=r, relation=r)
    return G


def get_truth_paths_new(q_entity, a_entity, G: nx.MultiDiGraph, max_hops=4):
    starts_raw = q_entity if isinstance(q_entity, (list, tuple, set)) else [q_entity]
    goals_raw = a_entity if isinstance(a_entity, (list, tuple, set)) else [a_entity]
    starts = {str(s) for s in starts_raw}
    goals = {str(t) for t in goals_raw}
    starts &= set(G.nodes)
    goals &= set(G.nodes)

    if not starts or not goals:
        return []

    dist = {}
    parents = defaultdict(set)   # parents[v] = {(u, r), ...}
    q = deque()

    for s in starts:
        dist[s] = 0
        q.append(s)

    min_goal_dist = None

    while q:
        u = q.popleft()
        du = dist[u]

        if (min_goal_dist is not None and du >= min_goal_dist) or du >= max_hops:
            continue

        for _, v, key, data in G.out_edges(u, keys=True, data=True):
            r = data.get('relation', key)
            if v not in dist:
                dist[v] = du + 1
                parents[v].add((u, r))
                q.append(v)
            elif dist[v] == du + 1:
                parents[v].add((u, r))

            if v in goals:
                if min_goal_dist is None or dist[v] < min_goal_dist:
                    min_goal_dist = dist[v]

    if min_goal_dist is None:
        return []

    paths = []

    def backtrack(v, acc):
        if v in starts:
            paths.append(list(reversed(acc)))
            return
        for (u, r) in parents[v]:
            if dist.get(u, 1e9) == dist[v] - 1:
                backtrack(u, acc + [(u, r, v)])

    for g in goals:
        if dist.get(g, 1e9) == min_goal_dist:
            backtrack(g, [])

    return paths
    
def get_simple_paths(q_entity: list, a_entity: list, graph: nx.Graph, hop=2) -> list:
    '''
    Get all simple paths connecting question and answer entities within given hop
    '''
    # Select paths
    paths = []
    for h in q_entity:
        if h not in graph:
            continue
        for t in a_entity:
            if t not in graph:
                continue
            try:
                for p in nx.all_simple_edge_paths(graph, h, t, cutoff=hop):
                    paths.append(p)
            except:
                pass
    # Add relation to paths
    result_paths = []
    for p in paths:
        result_paths.append([(e[0], graph[e[0]][e[1]]['relation'], e[1]) for e in p])
    return result_paths

def get_negative_paths(q_entity: list, a_entity: list, graph: nx.Graph, n_neg: int, hop=2) -> list:
    '''
    Get negative paths for question witin hop
    '''
    # sample paths
    start_nodes = []
    end_nodes = []
    node_idx = list(graph.nodes())
    for h in q_entity:
        if h in graph:
            start_nodes.append(node_idx.index(h))
    for t in a_entity:
        if t in graph:
            end_nodes.append(node_idx.index(t))
    paths = walker.random_walks(graph, n_walks=n_neg, walk_len=hop, start_nodes=start_nodes, verbose=False)
    # Add relation to paths
    result_paths = []
    for p in paths:
        tmp = []
        # remove paths that end with answer entity
        if p[-1] in end_nodes:
            continue
        for i in range(len(p)-1):
            u = node_idx[p[i]]
            v = node_idx[p[i+1]]
            tmp.append((u, graph[u][v]['relation'], v))
        result_paths.append(tmp)
    return result_paths

def get_random_paths(q_entity: list, graph: nx.Graph, n=3, hop=2) -> tuple [list, list]:
    '''
    Get negative paths for question witin hop
    '''
    # sample paths
    start_nodes = []
    node_idx = list(graph.nodes())
    for h in q_entity:
        if h in graph:
            start_nodes.append(node_idx.index(h))
    paths = walker.random_walks(graph, n_walks=n, walk_len=hop, start_nodes=start_nodes, verbose=False)
    # Add relation to paths
    result_paths = []
    rules = []
    for p in paths:
        tmp = []
        tmp_rule = []
        for i in range(len(p)-1):
            u = node_idx[p[i]]
            v = node_idx[p[i+1]]
            tmp.append((u, graph[u][v]['relation'], v))
            tmp_rule.append(graph[u][v]['relation'])
        result_paths.append(tmp)
        rules.append(tmp_rule)
    return result_paths, rules

def get_type_pairs_from_relations(rel_path, rel2edges):

    if not rel_path:
        return set()

    candidates = set(rel2edges.get(rel_path[0], set()))
    if not candidates:
        return set()

    for r in rel_path[1:]:
        edges = rel2edges.get(r, set())
        if not edges:
            return set()

        head_to_tails = defaultdict(set)
        for h, t in edges:
            head_to_tails[h].add(t)

        composed = set()
        for h0, mid in candidates:
            for t_next in head_to_tails.get(mid, set()):
                composed.add((h0, t_next))

        if not composed:
            return set()
        candidates = composed

    return candidates

def get_tail_types_from_relations(rel_path, rel2edges):
    if not rel_path:
        return set()

    first_rel_edges = rel2edges.get(rel_path[0], set())
    if not first_rel_edges:
        return set()
    current_tails = {t for (_, t) in first_rel_edges}

    for r in rel_path[1:]:
        edges = rel2edges.get(r, set())
        if not edges:
            return set()

        head_to_tails = defaultdict(set)
        for h, t in edges:
            head_to_tails[h].add(t)

        next_tails = set()
        for mid_type in current_tails:
            next_tails |= head_to_tails.get(mid_type, set())

        if not next_tails:
            return set()

        current_tails = next_tails

    return current_tails

def search_paths(start_entity, relation_path, index, max_depth=20):
    result_paths = []

    def dfs(entity, depth, path_so_far, visited_entities):
        if not isinstance(entity, str) or depth > max_depth:
            return
        if entity in visited_entities:
            return
        if depth == len(relation_path):
            result_paths.append(list(path_so_far))
            return

        expected_relation = relation_path[depth]
        neighbors = index.get(entity, {}).get(expected_relation, [])

        visited_entities.add(entity)
        for target in neighbors:
            path_so_far.append((entity, expected_relation, target))
            dfs(target, depth + 1, path_so_far, visited_entities)
            path_so_far.pop()
        visited_entities.remove(entity)

    dfs(start_entity, 0, [], set())
    return result_paths

def find_matching_paths_multirel(
    ontology_graph: nx.MultiDiGraph,
    head_type: str,
    tail_type: str,
    max_depth: int,
    forbid_node_repeat: bool = False
):
    paths = set()

    def pack(path_edges):
        return tuple((u, r, v)       for    (u, r, v, _key) in path_edges)

    def dfs(path, visited_edges, visited_nodes, depth):
        current = path[-1][2] if path else head_type

        if depth > max_depth:
            return

        if current == tail_type and path:
            paths.add(pack(path))
            return

        if depth == max_depth:
            return

        for _, nbr, key, data in ontology_graph.out_edges(current, keys=True, data=True):
            r = data.get('relation', key)
            if r is None:
                continue
            edge_id = (current, key, nbr)
            if edge_id in visited_edges:
                continue
            if forbid_node_repeat and nbr in visited_nodes:
                continue

            visited_edges.add(edge_id)
            if forbid_node_repeat:
                visited_nodes.add(nbr)
            path.append((current, str(r), nbr, key))

            dfs(path, visited_edges, visited_nodes, depth + 1)

            path.pop()
            if forbid_node_repeat:
                visited_nodes.remove(nbr)
            visited_edges.remove(edge_id)

    dfs([], set(), {head_type} if forbid_node_repeat else set(), 0)
    return list(paths)
def get_type_from_relations(rel_path, ontology_triples):
    if not rel_path:
        return []

    first_rel = rel_path[0]
    last_rel = rel_path[-1]

    head_types = set()
    tail_types = set()

    for h, r, t in ontology_triples:
        if r == first_rel:
            head_types.add(h)
        if r == last_rel:
            tail_types.add(t)

    type_pairs = []
    for h in head_types:
        for t in tail_types:
            type_pairs.append((h, t))

    return type_pairs


def get_paths_from_relations(rel_sep, rel2pair):
    if not rel_sep:
        return []

    rels = [str(r).strip() for r in (list(rel_sep) if isinstance(rel_sep, (list, tuple)) else [rel_sep])]
    first_rel = rels[0]
    if first_rel not in rel2pair:
        return []

    head, tail = rel2pair[first_rel]
    seq = [head, first_rel, tail]
    cur_type = tail

    for r in rels[1:]:
        pair = rel2pair.get(r)
        if pair is None:
            return []
        h, t = pair
        if h != cur_type:
            return []
        seq += [r, t]
        cur_type = t

    return [seq]

def get_paths_from_relations_multi(rel_sep, rel2types, beam_size=200, dedup=True):
    if not rel_sep:
        return []

    rels = [str(r).strip() for r in (list(rel_sep) if isinstance(rel_sep, (list, tuple)) else [rel_sep])]
    first = rels[0]
    pairs0 = rel2types.get(first, [])
    if not pairs0:
        return []

    partial = []
    for h, t in pairs0:
        partial.append((t, [h, first, t]))  # (cur_type, seq)

    for r in rels[1:]:
        pairs = rel2types.get(r, [])
        if not pairs:
            return []
        nxt = []
        head2tails = defaultdict(list)
        for h, t in pairs:
            head2tails[h].append(t)

        for cur_type, seq in partial:
            for t2 in head2tails.get(cur_type, []):
                nxt.append((t2, seq + [r, t2]))

        if not nxt:
            return []
        if len(nxt) > beam_size:
            nxt = nxt[:beam_size]
        partial = nxt

    out = [seq for _, seq in partial]
    if dedup:
        seen = set()
        uniq = []
        for s in out:
            key = tuple(s)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(s)
        out = uniq
    return out