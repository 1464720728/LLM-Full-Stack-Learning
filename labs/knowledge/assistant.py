"""Capstone 1: local evidence assistant. Explicit commands; never opens external URLs."""
import argparse
import json
from .store import CorpusStore,Context
from .fixtures import seed
from .rag import answer_report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db",default=":memory:")
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("seed",help="write fictional course fixtures")
    query=sub.add_parser("ask")
    query.add_argument("question")
    query.add_argument("--tenant",default="course")
    query.add_argument("--actor",default="learner")
    query.add_argument("--group",action="append",default=[])
    query.add_argument("--demo",action="store_true",help="seed fictional demo data before querying")
    args=parser.parse_args(argv)
    with CorpusStore(args.db) as store:
        if args.command=="seed":
            seed(store);print(json.dumps(store.counts(),sort_keys=True))
        else:
            if args.demo:seed(store)
            ctx=Context(args.tenant,args.actor,tuple(args.group))
            print(json.dumps(answer_report(store,ctx,args.question),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
