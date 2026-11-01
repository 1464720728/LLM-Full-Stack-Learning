"""Capstone 2: explicit durable learner memory. No hidden inference of preferences."""
import argparse
import json
from .memory import MemoryStore,select_memory
from .text import tokenize


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db",default=":memory:")
    parser.add_argument("--tenant",default="fictional-course")
    parser.add_argument("--owner",default="demo-learner")
    sub=parser.add_subparsers(dest="command",required=True)
    write=sub.add_parser("remember")
    write.add_argument("key");write.add_argument("value");write.add_argument("--evidence",required=True)
    write.add_argument("--expected",type=int,required=True)
    show=sub.add_parser("recall");show.add_argument("query",nargs="?",default="")
    remove=sub.add_parser("forget");remove.add_argument("key")
    sub.add_parser("demo")
    args=parser.parse_args(argv)
    with MemoryStore(args.db) as store:
        if args.command=="remember":
            rev=store.remember(args.tenant,args.owner,args.key,args.value,args.evidence,expected_revision=args.expected)
            print(json.dumps({"revision":rev}))
        elif args.command=="forget":
            store.forget(args.tenant,args.owner,args.key);print('{"forgotten":true}')
        elif args.command=="recall":
            memories=store.recall(args.tenant,args.owner)
            result=select_memory(memories,set(tokenize(args.query))) if args.query else [m.__dict__ for m in memories]
            print(json.dumps(result,ensure_ascii=False,indent=2))
        else:
            # A complete correction/retrieval/forget lifecycle, using a temporary namespace.
            key="demo.preferred_style"
            store.forget(args.tenant,args.owner,key)
            r1=store.remember(args.tenant,args.owner,key,"short examples","fictional user message #1",expected_revision=0)
            r2=store.remember(args.tenant,args.owner,key,"worked numeric examples","fictional correction #2",expected_revision=r1)
            print(json.dumps({"revisions":[r1,r2],"active":[m.__dict__ for m in store.recall(args.tenant,args.owner)]},ensure_ascii=False,indent=2))
            store.forget(args.tenant,args.owner,key)
            print(json.dumps({"history_after_forget":store.history(args.tenant,args.owner,key)}))


if __name__=="__main__":main()
