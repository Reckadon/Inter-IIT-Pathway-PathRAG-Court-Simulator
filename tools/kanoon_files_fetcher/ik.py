import logging
import http.client
import json
import urllib.parse


class IKApi:
    def __init__(self, args, storage):
        self.logger = logging.getLogger('ikapi')
        self.headers = {'Authorization': f'Token {args.token}', 'Accept': 'application/json'}
        self.basehost = 'api.indiankanoon.org'
        self.storage = storage
        self.maxcites = args.maxcites
        self.maxcitedby = args.maxcitedby
        self.orig = args.orig
        self.maxpages = min(args.maxpages, 100)  # Limit maxpages to 100
        self.pathbysrc = args.pathbysrc

    def call_api(self, url):
        connection = http.client.HTTPSConnection(self.basehost)
        connection.request('POST', url, headers=self.headers)
        response = connection.getresponse()
        return response.read()

    def fetch_doc(self, docid):
        url = f'/doc/{docid}/'
        args = []
        if self.maxcites > 0:
            args.append(f'maxcites={self.maxcites}')
        if self.maxcitedby > 0:
            args.append(f'maxcitedby={self.maxcitedby}')
        if args:
            url += '?' + '&'.join(args)
        return self.call_api(url)

    def search(self, q, pagenum, maxpages):
        q = urllib.parse.quote_plus(q.encode('utf8'))
        url = f'/search/?formInput={q}&pagenum={pagenum}&maxpages={maxpages}'
        return self.call_api(url)

    def save_search_results(self, q, max_docs=None):
        """
        Save search results for a given query.

        Args:
            q (str): The query string (keyword).
            max_docs (int): Maximum number of documents to fetch for this query.

        Returns:
            list: List of document IDs fetched.
        """
        keyword_dir = self.storage.get_keyword_path(q)  # Get directory for the keyword
        tocwriter = self.storage.get_tocwriter(keyword_dir)

        pagenum = 0
        current = 1
        docids = []

        while True:
            results = self.search(q, pagenum, self.maxpages)
            obj = json.loads(results)

            if 'docs' not in obj or len(obj['docs']) <= 0:
                break

            docs = obj['docs']
            self.logger.warning(f'Num results: {len(docs)}, pagenum: {pagenum}')

            for doc in docs:
                if max_docs and len(docids) >= max_docs:
                    self.logger.info(f"Reached max_docs limit ({max_docs}) for query '{q}'.")
                    return docids

                docid = doc['tid']
                toc = {'docid': docid, 'title': doc['title'], 'position': current,
                       'date': doc['publishdate'], 'court': doc['docsource']}
                tocwriter.writerow(toc)

                docpath = self.storage.get_docpath(keyword_dir, doc['docsource'], doc['publishdate'])
                if self.download_doc(docid, docpath):
                    docids.append(docid)
                current += 1

            pagenum += self.maxpages
        return docids

    def download_doc(self, docid, docpath):
        success = False
        orig_needed = self.orig
        jsonpath, origpath = self.storage.get_json_orig_path(docpath, docid)

        if not self.storage.exists(jsonpath):
            jsonstr = self.fetch_doc(docid)
            d = json.loads(jsonstr)
            if 'errmsg' in d:
                return success

            self.logger.info(f'Saved {d["title"]}')
            self.storage.save_json(jsonstr, jsonpath)
            success = True

            if orig_needed and not d.get('courtcopy'):
                orig_needed = False

        if orig_needed and not self.storage.exists_original(origpath):
            orig = self.fetch_orig_doc(docid)
            if orig:
                self.logger.info(f'Saved Original {d["title"]}')
                self.storage.save_original(orig, origpath)

        return success

    def fetch_orig_doc(self, docid):
        url = f'/origdoc/{docid}/'
        return self.call_api(url)
