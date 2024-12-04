import logging
import http.client
import json
import ssl
import time
import urllib.parse


class IKApi:
    def _init_(self, args, storage):
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
        max_retries = 3
        for attempt in range(max_retries):
            connection = None
            try:
                connection = http.client.HTTPSConnection(self.basehost)
                connection.request('POST', url, headers=self.headers)
                response = connection.getresponse()
                return response.read()
            
            except ssl.SSLError as e:
                self.logger.error(f"SSL error during API call: {e}")
                if attempt < max_retries - 1:
                    self.logger.info("Retrying after SSL error...")
                    time.sleep(2)
                else:
                    return None
            except http.client.HTTPException as e:
                self.logger.error(f"HTTP error during API call: {e}")
                if attempt < max_retries - 1:
                    self.logger.info("Retrying after HTTP error...")
                    time.sleep(2)
                else:
                    return None
            except Exception as e:
                self.logger.error(f"Unexpected error during API call: {e}")
                return None

    def fetch_doc(self, docid):
        url = f'/doc/{docid}/'
        args = []
        if self.maxcites > 0:
            args.append(f'maxcites={self.maxcites}')
        if self.maxcitedby > 0:
            args.append(f'maxcitedby={self.maxcitedby}')
        if args:
            url += '?' + '&'.join(args)

        try:
            response = self.call_api(url)
            if not response:
                self.logger.error(f"Failed to fetch document for docid {docid}. No response received.")
                return None
            return response
        except Exception as e:
            self.logger.error(f"An error occurred while fetching document for docid {docid}: {e}")
            return None


    def search(self, q, pagenum, maxpages):
        q = urllib.parse.quote_plus(q.encode('utf8'))
        url = f'/search/?formInput={q}&pagenum={pagenum}&maxpages={maxpages}'
        try:
            response = self.call_api(url)
            if not response:
                self.logger.error(f"Search API returned no response for query '{q}', page {pagenum}.")
                return None
            return response
        except Exception as e:
            self.logger.error(f"An error occurred during the search for query '{q}', page {pagenum}: {e}")
            return None


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
            try:
                results = self.search(q, pagenum, self.maxpages)
                if not results:
                    self.logger.warning(f"No results returned for query '{q}' on page {pagenum}.")
                    break
                
                try:
                    obj = json.loads(results)
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to decode JSON for query '{q}' on page {pagenum}: {e}")
                    break

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
            except Exception as e:
                self.logger.error(f"An error occurred while fetching results for query '{q}': {e}")
                break
        return docids

    def download_doc(self, docid, docpath):
        success = False
        orig_needed = self.orig
        jsonpath, origpath = self.storage.get_json_orig_path(docpath, docid)

        if not self.storage.exists(jsonpath):
            try:
                jsonstr = self.fetch_doc(docid)
                if not jsonstr:
                    self.logger.error(f"No response received for docid {docid}.")
                    return success

                try:
                    d = json.loads(jsonstr)
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to decode JSON for docid {docid}: {e}")
                    return success

                if 'errmsg' in d:
                    self.logger.warning(f"Error message received for docid {docid}: {d['errmsg']}")
                    return success

                self.logger.info(f'Saved {d["title"]}')
                self.storage.save_json(jsonstr, jsonpath)
                success = True

                if orig_needed and not d.get('courtcopy'):
                    orig_needed = False

            except Exception as e:
                self.logger.error(f"An error occurred while fetching or processing docid {docid}: {e}")
                return success

        if orig_needed and not self.storage.exists_original(origpath):
            try:
                orig = self.fetch_orig_doc(docid)
                if orig:
                    self.logger.info(f'Saved Original {d["title"]}')
                    self.storage.save_original(orig, origpath)
            except Exception as e:
                self.logger.error(f"An error occurred while fetching or saving the original doc for docid {docid}: {e}")

        return success


    def fetch_orig_doc(self, docid):
        url = f'/origdoc/{docid}/'
        try:
            response = self.call_api(url)
            if not response:
                self.logger.error(f"Failed to fetch original document for docid {docid}. No response received.")
                return None
            return response
        except Exception as e:
            self.logger.error(f"An error occurred while fetching original document for docid {docid}: {e}")
            return None