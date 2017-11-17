#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<algorithm>
#include<cmath>
#include<cstdlib>
using namespace std;

bool debug=false;


string version;
string trainortest = "test";

map<string,int> relation2id,entity2id;
map<int,string> id2entity,id2relation;
map<string,string> mid2name,mid2type;
map<int,map<int,int> > entity2num;
map<int,int> e2num;
map<pair<string,string>,map<string,double> > rel_left,rel_right;

int relation_num,entity_num;
int hits_margin1 = 1;
int hits_margin2 = 3;
int hits_margin3 = 10;
int hits_margin4 = 100;

double sigmod(double x)
{
    return 1.0/(1+exp(-x));
}

double vec_len(vector<double> a)
{
    double res=0;
    for (int i=0; i<a.size(); i++)
        res+=a[i]*a[i];
    return sqrt(res);
}

void vec_output(vector<double> a)
{
    for (int i=0; i<a.size(); i++)
    {
        cout<<a[i]<<"\t";
        if (i%10==9)
            cout<<endl;
    }
    cout<<"-------------------------"<<endl;
}

double sqr(double x)
{
    return x*x;
}

char buf[100000],buf1[100000];

int my_cmp(pair<double,int> a,pair<double,int> b)
{
    return a.first>b.first;
}

double cmp(pair<int,double> a, pair<int,double> b)
{
    return a.second < b.second;
}

class Test{
    vector<vector<double> > relation_vec, entity_vec;

    vector<int> h,l,r;
    vector<int> fb_h,fb_l,fb_r;
    vector<int> ranks;
    map<pair<int,int>, map<int,int> > ok;
    double res ;
public:
    void add(int x,int y,int z, bool flag)
    {
        if (flag)
        {
            fb_h.push_back(x);
            fb_r.push_back(z);
            fb_l.push_back(y);
        }
        ok[make_pair(x,z)][y]=1;
    }
    double norm(vector<double> &a)    //Normalize the vector
    {
        double x = vec_len(a);
        if (x>1)
        for (int ii=0; ii<a.size(); ii++)
                a[ii]/=x;
        return 0;
    }

    int rand_max(int x)
    {
        int res = (rand()*rand())%x;
        if (res<0)
            res+=x;
        return res;
    }
    double len;


    void run()
    {
        cout<<relation_num<<' '<<entity_num<<endl;
        int relation_num_fb=relation_num;

        double lsum=0 ,lsum_filter= 0;
        double rsum = 0,rsum_filter=0;
        double lp_n=0,lp_n_filter1, lp_n_filter2, lp_n_filter3, lp_n_filter4;
        double rp_n=0,rp_n_filter1, rp_n_filter2, rp_n_filter3, rp_n_filter4;
        double mrr_left=0, mrr_right=0;
        map<int,double> lsum_r,lsum_filter_r1, lsum_filter_r2, lsum_filter_r3, lsum_filter_r4;
        map<int,double> rsum_r,rsum_filter_r1, rsum_filter_r2, rsum_filter_r3, rsum_filter_r4;
        map<int,double> lp_n_r,lp_n_filter_r1, lp_n_filter_r2, lp_n_filter_r3, lp_n_filter_r4;
        map<int,double> rp_n_r,rp_n_filter_r1, rp_n_filter_r2, rp_n_filter_r3, rp_n_filter_r4;
    
/*        FILE* f1 = fopen("valid_head_pred.txt","r");
        vector<vector<pair<int,double> > > a;
        a.resize(fb_h.size());
        for (int i=0; i<a.size();i++)
        {
            a[i].resize(entity_num);
            for (int ii=0; ii<entity_num; ii++)
            {
                fscanf(f1,"%lf",&a[i][ii].second);
                a[i][ii].first = ii;
            }
        }

        fclose(f1);

        for (int testid = 0; testid<fb_l.size(); testid+=1)
        {
            int h = fb_h[testid];
            int l = fb_l[testid];
            int rel = fb_r[testid];
            double ttt=0;
            int filter = 0;

            //ascending order sort
            sort(a[testid].begin(),a[testid].end(),cmp);

            for (int i=a[testid].size()-1; i>=0; i--)
            {
                if (ok[make_pair(a[testid][i].first,rel)].count(l)>0)
                    ttt++;
                if (ok[make_pair(a[testid][i].first,rel)].count(l)==0)
                    filter+=1;
                if (a[testid][i].first ==h)
                {
                    lsum+=a[testid].size()-i;
                    lsum_filter += filter+1;
                    mrr_left += 1.0/(filter+1);
                    lsum_r[rel]+=a[testid].size()-i;
                    lsum_filter_r[rel]+=filter+1;
                    if (a[testid].size()-i<=hits_margin)
                    {
                        lp_n+=1;
                        lp_n_r[rel]+=1;
                    }
                    if (filter<hits_margin1)
                    {
                        lp_n_filter1+=1;
                        lp_n_filter_r[rel]+=1;
                    }
                    if (filter<hits_margin2)
                    {
                        lp_n_filter2+=1;
                        lp_n_filter_r[rel]+=1;
                    }
                    if (filter<hits_margin3)
                    {
                        lp_n_filter3+=1;
                        lp_n_filter_r[rel]+=1;
                    }
                    if (filter<hits_margin4)
                    {
                        lp_n_filter4+=1;
                        lp_n_filter_r[rel]+=1;
                    }
                    break;
                }
            }
        }*/
        vector<vector<pair<int,double> > > a;
        a.resize(fb_h.size());
    
        FILE* f3 = fopen("valid_tail_pred.txt","r");
        for (int i=0; i<a.size();i++)
        {
            a[i].resize(entity_num);
            for (int ii=0; ii<entity_num; ii++)
            { 
                fscanf(f3,"%lf",&a[i][ii].second);
                a[i][ii].first = ii;
            }
        }
        fclose(f3);

        ranks.resize(entity_num);

        for(int testid = 0; testid<fb_l.size(); testid+=1)
        {
            int h = fb_h[testid];
            int l = fb_l[testid];
            int rel = fb_r[testid];            
            
            sort(a[testid].begin(),a[testid].end(),cmp);

            double ttt=0;
            int filter = 0;
            for (int i=a[testid].size()-1; i>=0; i--)
            {
                if (ok[make_pair(h,rel)].count(a[testid][i].first)>0)
                    ttt++;
                if (ok[make_pair(h,rel)].count(a[testid][i].first)==0)
                    filter+=1;
                if (a[testid][i].first==l)
                {
                    ranks[filter+1]++;
                    rsum+=a[testid].size()-i;
                    rsum_filter+=filter+1;
                    mrr_right += 1.0/(filter+1);
                    rsum_r[rel]+=a[testid].size()-i;
                    //rsum_filter_r[rel]+=filter+1;
/*                  if (a[testid].size()-i<=hits_margin)
                    {
                        rp_n+=1;
                        rp_n_r[rel]+=1;
                    }*/
                    if (filter<hits_margin1)
                    {
                        rp_n_filter1+=1;
                        //rp_n_filter_r[rel]+=1;
                    }
                    if (filter<hits_margin2)
                    {
                        rp_n_filter2+=1;
                        //rp_n_filter_r[rel]+=1;
                    }
                    if (filter<hits_margin3)
                    {
                        rp_n_filter3+=1;
                        //rp_n_filter_r[rel]+=1;
                    }
                    if (filter<hits_margin4)
                    {
                        rp_n_filter4+=1;
                        //rp_n_filter_r[rel]+=1;
                    }
                    break;
                }
            }
        }
        /*cout<<"left:"<<lsum/fb_l.size()<<'\t'<<lp_n/fb_l.size()<<"\t"<<lsum_filter/fb_l.size()<<'\t'<<lp_n_filter/fb_l.size()<<endl;
        cout<<"right:"<<rsum/fb_r.size()<<'\t'<<rp_n/fb_r.size()<<'\t'<<rsum_filter/fb_r.size()<<'\t'<<rp_n_filter/fb_r.size()<<endl;
        cout<<"MRR left: " << mrr_left/fb_h.size() << ", MRR right: " << mrr_right/fb_h.size();*/

/*      cout << lsum_filter/fb_l.size()<<'\t'<<lp_n_filter3/fb_l.size()<<endl;
        cout << rsum_filter/fb_r.size()<<'\t'<<rp_n_filter3/fb_r.size()<<endl;*/
        cout << "Hits@1    : " << rp_n_filter1/fb_r.size() << endl;
        cout << "Hits@3    : " << rp_n_filter2/fb_r.size() << endl;
        cout << "Hits@10   : " << rp_n_filter3/fb_r.size() << endl;
        cout << "Hits@100  : " << rp_n_filter4/fb_r.size() << endl;
        cout << "Mean rank : " << rsum_filter/fb_r.size() << endl;
        cout << "MRR       : " << mrr_right/fb_h.size() << endl;        

        FILE* f9 = fopen("ranks.txt","w");
        for (int i=0; i<ranks.size();i++)
        {
            fprintf(f9,"%d ",ranks[i]);
        }
        fclose(f9);

    }

};
Test test;

void prepare()
{
    FILE* f1 = fopen("../data/entity2id.txt","r");
    FILE* f2 = fopen("../data/relation2id.txt","r");
    int x;
    while (fscanf(f1,"%s%d",buf,&x)==2)
    {
        string st=buf;
        entity2id[st]=x;
        id2entity[x]=st;
        mid2type[st]="None";
        entity_num++;
    }
    while (fscanf(f2,"%s%d",buf,&x)==2)
    {
        string st=buf;
        relation2id[st]=x;
        id2relation[x]=st;
        relation_num++;
    }
    FILE* f_kb = fopen("../data/test.txt","r");
    while (fscanf(f_kb,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb,"%s",buf);
        string s3=buf;
        fscanf(f_kb,"%s",buf);
        string s2=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            cout<<"miss relation:"<<s3<<endl;
            relation2id[s3] = relation_num;
            relation_num++;
        }
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],false);
    }
    fclose(f_kb);
    FILE* f_kb1 = fopen("../data/train.txt","r");
    while (fscanf(f_kb1,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb1,"%s",buf);
        string s3=buf;
        fscanf(f_kb1,"%s",buf);
        string s2=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            relation2id[s3] = relation_num;
            relation_num++;
        }

        entity2num[relation2id[s3]][entity2id[s1]]+=1;
        entity2num[relation2id[s3]][entity2id[s2]]+=1;
        e2num[entity2id[s1]]+=1;
        e2num[entity2id[s2]]+=1;
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],false);
    }
    fclose(f_kb1);
    FILE* f_kb2 = fopen("../data/valid.txt","r");
    while (fscanf(f_kb2,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb2,"%s",buf);
        string s3=buf;
        fscanf(f_kb2,"%s",buf);
        string s2=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            relation2id[s3] = relation_num;
            relation_num++;
        }
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],true);
    }
    fclose(f_kb2);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc,char**argv)
{
    int i;
    //if ((i = ArgPos((char *)"-hits", argc, argv)) > 0) hits_margin = atoi(argv[i + 1]);
    prepare();
    test.run();
}

