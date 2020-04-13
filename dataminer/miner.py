import re
import requests
from datetime import datetime, timedelta

import os
import copy
import json

from utils import clone_repo

QUERY = """
{
    search(query: "is:public mirror:false archived:false created:DATE_FROM..DATE_TO", type: REPOSITORY, first: 50 AFTER) {
        repositoryCount
        pageInfo {
            endCursor
            startCursor
            hasNextPage
        }
        edges {
            node {
                ... on Repository {
                    id
                    owner { login }
                    name
                    url
                    createdAt
                    description
                    primaryLanguage { name }
                    object(expression: "master:") {
                        ... on Tree {
                            entries {
                                name
                                type
                            }
                        }
                    }
                }
            }
        }
    }
    rateLimit {
        limit
        cost
        remaining
        resetAt
    }
}
"""


class GithubMiner():

    def __init__(self, 
                 date_from: datetime, 
                 date_to: datetime
                ):

        self.date_from = date_from.strftime('%Y-%m-%dT%H:%M:%SZ') 
        self.date_to = date_to.strftime('%Y-%m-%dT%H:%M:%SZ')

        self.quota = 0
        self.quota_reset_at = None

        self.query = re.sub('DATE_FROM', str(self.date_from), QUERY) 
        self.query = re.sub('DATE_TO', str(self.date_to), self.query) 

    def run_query(self, query): 
        """
        Run a graphql query 
        """
        request = requests.post('https://api.github.com/graphql', json={'query': query}, headers={'Authorization': 'token 0d0ff8e340e1c6cdaa9aacbff6206b55cbbb63f4'})
        
        if request.status_code == 200:
            return request.json()
        
        else:
            print("Query failed to run by returning code of {}. {}".format(request.status_code, query))
            
            with open("logs/failed_queries.txt", "a+") as file:
                file.write(f'{self.date_from} {self.date_to} \n')

            return None

    def filter_repositories(self, edges):

        for node in edges:
            
            node = node.get('node')

            if not node:
                continue

            object = node.get('object')
            if not object:
                continue
            
            dirs = [entry.get('name') for entry in object.get('entries', []) if entry.get('type') == 'tree']

            yield dict(
                    id=node.get('id'),
                    default_branch=node.get('defaultBranchRef', {}).get('name'),
                    owner=node.get('owner', {}).get('login', ''),
                    name=node.get('name', ''),
                    url=node.get('url'),
                    description=node['description'] if node['description'] else '',
                    dirs=dirs,
                    createdAt=node.get('createdAt', 'No date found')
            )

    def mine(self):
        
        has_next_page = True
        end_cursor = None

        while has_next_page:
            
            tmp_query = re.sub('AFTER', '', self.query) if not end_cursor else re.sub('AFTER', f', after: "{end_cursor}"', self.query)
            
            result = self.run_query(tmp_query)

            if not result:
                break
            
            if not result.get('data'):
                break

            if not result['data'].get('search'):
                break
            
            self.quota = int(result['data']['rateLimit']['remaining'])
            self.quota_reset_at = result['data']['rateLimit']['resetAt']

            has_next_page = bool(result['data']['search']['pageInfo'].get('hasNextPage'))
            end_cursor = str(result['data']['search']['pageInfo'].get('endCursor'))

            edges = result['data']['search'].get('edges', [])

            for repo in self.filter_repositories(edges):
                yield repo


def main(date_from, date_to):
    
    github_miner = GithubMiner(
        date_from=date_from,
        date_to=date_to
    )
    
    i = 0 
    
    for repo in github_miner.mine():
    
        i += 1

        if (re.search(r"\btosca\b", repo['description'].lower()) or re.search(r"\btosca\b", repo['owner'].lower()) or re.search(r"\btosca\b", repo['name'].lower())):
            clone_repo(repo['owner'], repo['name'])

        else:
            continue
    
    
    print(f'{i} repositories mined')
    print(f'Quota: {github_miner.quota}')
    print(f'Quota will reset at: {github_miner.quota_reset_at}')
    print('---------------')


    with open("logs/executed_queries.txt", "a+") as file:
        file.write(f'mined: {i} from: {date_from} to: {date_to} \n')


if __name__=='__main__':
    date_from = datetime.strptime('2014-03-27 00:00:00', '%Y-%m-%d %H:%M:%S')
    date_to = datetime.strptime('2014-03-27 12:00:00', '%Y-%m-%d %H:%M:%S')
    now = datetime.strptime('2020-03-31 00:00:00', '%Y-%m-%d %H:%M:%S')

    while date_to <= now:
        print(f'Searching for: {date_from}..{date_to}. Analysis started at {str(datetime.now())}')
        main(date_from, date_to)
        date_from = date_to
        date_to += timedelta(hours=12)
