#! /usr/bin/env python3
#
# This file is part of pyperplan.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#

# TODO: Give searches and heuristics commandline options and reenable preferred
# operators.

import argparse
import logging
import os
import re
import subprocess
import sys
import time

import grounding
import heuristics
from pddl.parser import Parser
import search
import tools
import json



SEARCHES = {
    "astar": search.astar_search,
    "wastar": search.weighted_astar_search,
    "gbf": search.greedy_best_first_search,
    "bfs": search.breadth_first_search,
    "ehs": search.enforced_hillclimbing_search,
    "ids": search.iterative_deepening_search,
    "sat": search.sat_solve,
}
PDDL_GLOBAL_PATH = os.path.dirname(os.getcwd())
DEFAULT_DOMAIN = PDDL_GLOBAL_PATH + '/furniture_assembly_domain/domain.pddl'

NUMBER = re.compile(r"\d+")

with open(PDDL_GLOBAL_PATH + '/furniture_assembly_domain/name_mapping.json', 'r') as f:
    NAME_MAPPING = json.load(f)
    

def get_heuristics():
    """
    Scan all python modules in the "heuristics" directory for classes ending
    with "Heuristic".
    """
    heuristics = []
    src_dir = os.path.dirname(os.path.abspath(__file__))
    heuristics_dir = os.path.abspath(os.path.join(src_dir, "heuristics"))
    for filename in os.listdir(heuristics_dir):
        if not filename.endswith(".py"):
            continue
        module = tools.import_python_file(os.path.join(heuristics_dir, filename))
        heuristics.extend(
            [
                getattr(module, cls)
                for cls in dir(module)
                if cls.endswith("Heuristic")
                and cls != "Heuristic"
                and not cls.startswith("_")
            ]
        )
    return heuristics


def _get_heuristic_name(cls):
    name = cls.__name__
    assert name.endswith("Heuristic")
    return name[:-9].lower()


HEURISTICS = {_get_heuristic_name(heur): heur for heur in get_heuristics()}


def validator_available():
    return tools.command_available(["validate", "-h"])


def find_domain(problem):
    """
    This function tries to guess a domain file from a given problem file.
    It first uses a file called "domain.pddl" in the same directory as
    the problem file. If the problem file's name contains digits, the first
    group of digits is interpreted as a number and the directory is searched
    for a file that contains both, the word "domain" and the number.
    This is conforming to some domains where there is a special domain file
    for each problem, e.g. the airport domain.

    @param problem    The pathname to a problem file
    @return A valid name of a domain
    """
    dir, name = os.path.split(problem)
    number_match = NUMBER.search(name)
    number = number_match.group(0)
    domain = os.path.join(dir, "domain.pddl")
    for file in os.listdir(dir):
        if "domain" in file and number in file:
            domain = os.path.join(dir, file)
            break
    if not os.path.isfile(domain):
        logging.error('Domain file "{}" can not be found'.format(domain))
        sys.exit(1)
    logging.info("Found domain {}".format(domain))
    return domain


def _parse(domain_file, problem_file):
    # Parsing
    parser = Parser(domain_file, problem_file)
    logging.info("Parsing Domain {}".format(domain_file))
    domain = parser.parse_domain()
    logging.info("Parsing Problem {}".format(problem_file))
    problem = parser.parse_problem(domain)
    logging.debug(domain)
    logging.info("{} Predicates parsed".format(len(domain.predicates)))
    logging.info("{} Actions parsed".format(len(domain.actions)))
    logging.info("{} Objects parsed".format(len(problem.objects)))
    logging.info("{} Constants parsed".format(len(domain.constants)))
    return problem


def _ground(problem):
    logging.info("Grounding start: {}".format(problem.name))
    task = grounding.ground(problem)
    logging.info("Grounding end: {}".format(problem.name))
    logging.info("{} Variables created".format(len(task.facts)))
    logging.info("{} Operators created".format(len(task.operators)))
    return task


def _search(task, search, heuristic, use_preferred_ops=False):
    logging.info("Search start: {}".format(task.name))
    if heuristic:
        if use_preferred_ops:
            solution = search(task, heuristic, use_preferred_ops)
        else:
            solution = search(task, heuristic)
    else:
        solution = search(task)
    logging.info("Search end: {}".format(task.name))
    return solution


def _write_solution(solution, filename):
    assert solution is not None
    with open(filename, "w") as file:
        for op in solution:
            print(op.name, file=file)


def search_plan(
    domain_file, problem_file, search, heuristic_class, use_preferred_ops=False
):
    """
    Parses the given input files to a specific planner task and then tries to
    find a solution using the specified  search algorithm and heuristics.

    @param domain_file      The path to a domain file
    @param problem_file     The path to a problem file in the domain given by
                            domain_file
    @param search           A callable that performs a search on the task's
                            search space
    @param heuristic_class  A class implementing the heuristic_base.Heuristic
                            interface
    @return A list of actions that solve the problem
    """
    problem = _parse(domain_file, problem_file)
    task = _ground(problem)
    heuristic = None
    if not heuristic_class is None:
        heuristic = heuristic_class(task)
    search_start_time = time.process_time()
    if use_preferred_ops and isinstance(heuristic, heuristics.hFFHeuristic):
        solution = _search(task, search, heuristic, use_preferred_ops=True)
    else:
        solution = _search(task, search, heuristic)
    logging.info("Search time: {:.2}".format(time.process_time() - search_start_time))
    return solution


def validate_solution(domain_file, problem_file, solution_file):
    if not validator_available():
        logging.info(
            "validate could not be found on the PATH so the plan can "
            "not be validated."
        )
        return

    cmd = ["validate", domain_file, problem_file, solution_file]
    exitcode = subprocess.call(cmd, stdout=subprocess.PIPE)

    if exitcode == 0:
        logging.info("Plan correct")
    else:
        logging.warning("Plan NOT correct")
    return exitcode == 0


def plan_sequence(obj, domain=DEFAULT_DOMAIN, search=SEARCHES['astar'], heuristic=HEURISTICS['hff']):
    problem = domain.replace('domain.pddl', obj + '_task.pddl')
    solution = search_plan(
        domain,
        problem,
        search,
        heuristic,
        use_preferred_ops=heuristic == "hffpo",
    )
    name_map = NAME_MAPPING[obj]
    action_sequence = []
    # print("PYPERPLAN SOLUTION: ", solution)
    for s in solution:
        action = s.name
        words = action.split(' ')
        if len(words) == 4:  # pick-up-from-floor robot-gripper part-handle part
            gripper = 'right' if words[1] == 'robot-gripper2' else 'left'
            action_sequence.append(['grasp', gripper, name_map[words[-1][:-1]]])
        elif len(words) == 6:  # screw-into/connect-to robot-gripper part1-site part1 part2-site part2
            move = 'screw' if words[0][1:] == 'screw-into' else 'insert'
            action_sequence.append([move, 'left', name_map[words[2]], name_map[words[4]]])
        elif len(words) == 7: # insert-into-base robot-gripper part1-site part1 part2-site part2 part3
            move = 'insert'
            action_sequence.append([move, 'left', name_map[words[2]], name_map[words[4]]])
        elif len(words) == 10: # connect-two-to-[something]
            move = 'connect2'
            if words[0][1:] == 'connect-two-to-stool': # connect-two-to-stool gripper1 part1-site1 part1 seat-site seat gripper2 part2-site part2 part1-site2
                action_sequence.append([move, 'left', name_map[words[2]], name_map[words[4]], 'right', name_map[words[7]], name_map[words[9][:-1]]])
            elif words[0][1:] == 'connect-two-to-seat': # connect-two-to-seat gripper1 part1-site part1 gripper2 part2-site part2 seat-site1 seat-site2 seat
                action_sequence.append([move, 'left', name_map[words[2]], name_map[words[7]], 'right', name_map[words[5]], name_map[words[8]]])
    return action_sequence
    

def main():
    # Commandline parsing
    log_levels = ["debug", "info", "warning", "error"]

    # get pretty print names for the search algorithms:
    # use the function/class name and strip off '_search'
    def get_callable_names(callables, omit_string):
        names = [c.__name__ for c in callables]
        names = [n.replace(omit_string, "").replace("_", " ") for n in names]
        return ", ".join(names)

    search_names = get_callable_names(SEARCHES.values(), "_search")

    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument(dest="domain", nargs="?")
    argparser.add_argument(dest="problem")
    argparser.add_argument("-l", "--loglevel", choices=log_levels, default="info")
    argparser.add_argument(
        "-H",
        "--heuristic",
        choices=HEURISTICS.keys(),
        help="Select a heuristic",
        default="hff",
    )
    argparser.add_argument(
        "-s",
        "--search",
        choices=SEARCHES.keys(),
        help="Select a search algorithm from {}".format(search_names),
        default="bfs",
    )
    args = argparser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper()),
        format="%(asctime)s %(levelname)-8s %(message)s",
        stream=sys.stdout,
    )

    hffpo_searches = ["gbf", "wastar", "ehs"]
    if args.heuristic == "hffpo" and args.search not in hffpo_searches:
        print(
            "ERROR: hffpo can currently only be used with %s\n" % hffpo_searches,
            file=sys.stderr,
        )
        argparser.print_help()
        exit(2)

    args.problem = os.path.abspath(args.problem)
    if args.domain is None:
        args.domain = find_domain(args.problem)
    else:
        args.domain = os.path.abspath(args.domain)

    search = SEARCHES[args.search]
    heuristic = HEURISTICS[args.heuristic]

    if args.search in ["bfs", "ids", "sat"]:
        heuristic = None

    logging.info("using search: %s" % search.__name__)
    logging.info("using heuristic: %s" % (heuristic.__name__ if heuristic else None))
    use_preferred_ops = args.heuristic == "hffpo"
    solution = search_plan(
        args.domain,
        args.problem,
        search,
        heuristic,
        use_preferred_ops=use_preferred_ops,
    )

    if solution is None:
        logging.warning("No solution could be found")
    else:
        solution_file = args.problem + ".soln"
        logging.info("Plan length: %s" % len(solution))
        _write_solution(solution, solution_file)
        validate_solution(args.domain, args.problem, solution_file)


if __name__ == "__main__":
    main()

# ./pyperplan/pyperplan.py -s astar furniture_assembly_domain/domain.pddl furniture_assembly_domain/table_lack_task.pddl