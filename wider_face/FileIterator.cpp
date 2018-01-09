#include "FileIterator.h"
using namespace std;

CFileIterator::CFileIterator(const std::string& path, const std::string filter /*= ""*/)
    : m_sPath(path)
    , m_sFilter(filter)
    , m_hFile(-1)
{

}

CFileIterator::~CFileIterator()
{

}

bool CFileIterator::FindNext()
{
    bool ret = false;
    if (m_hFile == -1)
    {
        string fullPath = m_sPath + "/" + m_sFilter;
        m_hFile = _findfirst(fullPath.c_str(), &m_fileAttribute);
        ret = m_hFile != -1;
    }
    else
    {
        ret = _findnext(m_hFile, &m_fileAttribute) == 0;
    }

    return ret;
}

std::string CFileIterator::FileName() const
{
    return m_fileAttribute.name;
}

std::string CFileIterator::FullFileName() const
{
    return m_sPath + "/" + m_fileAttribute.name;
}

