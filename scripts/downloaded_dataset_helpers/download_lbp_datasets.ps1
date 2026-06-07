param(
    [switch]$SkipObjects365
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\data\raw")).Path
$ArchiveRoot = Join-Path $Root "_downloads"

function Ensure-Dir($Path) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Download-File($Url, $OutFile, [Int64]$ExpectedBytes = 0) {
    Ensure-Dir (Split-Path -Parent $OutFile)
    if ((Test-Path -LiteralPath $OutFile) -and $ExpectedBytes -gt 0) {
        $actual = (Get-Item -LiteralPath $OutFile).Length
        if ($actual -eq $ExpectedBytes) {
            Write-Host "SKIP download complete: $OutFile"
            return
        }
    }
    if (Test-Path -LiteralPath $OutFile) {
        Write-Host "RESUME $Url"
    } else {
        Write-Host "DOWNLOAD $Url"
    }
    curl.exe -L --fail --retry 8 --retry-delay 10 --connect-timeout 30 -C - -o $OutFile $Url
}

function Download-FileSegmented($Url, $OutFile, [Int64]$ExpectedBytes, [int]$Parts = 8) {
    Ensure-Dir (Split-Path -Parent $OutFile)
    if ((Test-Path -LiteralPath $OutFile) -and ((Get-Item -LiteralPath $OutFile).Length -eq $ExpectedBytes)) {
        Write-Host "SKIP segmented download complete: $OutFile"
        return
    }

    if (Test-Path -LiteralPath $OutFile) {
        $actual = (Get-Item -LiteralPath $OutFile).Length
        if ($actual -ne $ExpectedBytes) {
            Write-Host "REMOVE incomplete final file before segmented download: $OutFile"
            Remove-Item -LiteralPath $OutFile -Force
        }
    }

    $partDir = "$OutFile.parts"
    Ensure-Dir $partDir
    $chunk = [Math]::Ceiling($ExpectedBytes / $Parts)
    $jobs = @()
    for ($i = 0; $i -lt $Parts; $i++) {
        $start = [Int64]($i * $chunk)
        $end = [Int64]([Math]::Min($ExpectedBytes - 1, (($i + 1) * $chunk) - 1))
        $partPath = Join-Path $partDir ("part{0:D2}" -f $i)
        $partExpected = $end - $start + 1
        if ((Test-Path -LiteralPath $partPath) -and ((Get-Item -LiteralPath $partPath).Length -eq $partExpected)) {
            Write-Host "SKIP complete part $i"
            continue
        }
        if (Test-Path -LiteralPath $partPath) {
            Remove-Item -LiteralPath $partPath -Force
        }
        $stdout = Join-Path $partDir ("part{0:D2}.out.log" -f $i)
        $stderr = Join-Path $partDir ("part{0:D2}.err.log" -f $i)
        Write-Host "DOWNLOAD part $i bytes $start-$end"
        $args = @("-L", "--fail", "--retry", "8", "--retry-delay", "10", "--connect-timeout", "30", "-r", "$start-$end", "-o", $partPath, $Url)
        $jobs += Start-Process -FilePath "curl.exe" -ArgumentList $args -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    }

    foreach ($job in $jobs) {
        $job.WaitForExit()
    }

    for ($i = 0; $i -lt $Parts; $i++) {
        $start = [Int64]($i * $chunk)
        $end = [Int64]([Math]::Min($ExpectedBytes - 1, (($i + 1) * $chunk) - 1))
        $partPath = Join-Path $partDir ("part{0:D2}" -f $i)
        $partExpected = $end - $start + 1
        if (-not (Test-Path -LiteralPath $partPath)) {
            throw "Missing segmented part: $partPath"
        }
        $actual = (Get-Item -LiteralPath $partPath).Length
        if ($actual -ne $partExpected) {
            throw "Wrong size for $partPath. Expected $partExpected, got $actual"
        }
    }

    Write-Host "COMBINE segmented file: $OutFile"
    $outStream = [System.IO.File]::Open($OutFile, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
    try {
        for ($i = 0; $i -lt $Parts; $i++) {
            $partPath = Join-Path $partDir ("part{0:D2}" -f $i)
            $inStream = [System.IO.File]::OpenRead($partPath)
            try {
                $inStream.CopyTo($outStream)
            } finally {
                $inStream.Dispose()
            }
        }
    } finally {
        $outStream.Dispose()
    }

    $finalSize = (Get-Item -LiteralPath $OutFile).Length
    if ($finalSize -ne $ExpectedBytes) {
        throw "Combined file has wrong size. Expected $ExpectedBytes, got $finalSize"
    }
    Remove-Item -LiteralPath $partDir -Recurse -Force
}

function Expand-ZipIfNeeded($Archive, $Dest, $CheckPath) {
    Ensure-Dir $Dest
    if (Test-Path -LiteralPath $CheckPath) {
        Write-Host "SKIP extract exists: $CheckPath"
        return
    }
    Write-Host "EXTRACT $Archive -> $Dest"
    tar.exe -xf $Archive -C $Dest
}

function Expand-TarIfNeeded($Archive, $Dest, $CheckPath) {
    Ensure-Dir $Dest
    if (Test-Path -LiteralPath $CheckPath) {
        Write-Host "SKIP extract exists: $CheckPath"
        return
    }
    Write-Host "EXTRACT $Archive -> $Dest"
    tar.exe -xf $Archive -C $Dest
}

Ensure-Dir $ArchiveRoot

# MS-COCO 2017 for OV-COCO and LVIS image reuse.
$cocoArchive = Join-Path $ArchiveRoot "coco"
$cocoRoot = Join-Path $Root "coco"
Ensure-Dir $cocoArchive
Ensure-Dir $cocoRoot
Download-FileSegmented "http://images.cocodataset.org/zips/train2017.zip" (Join-Path $cocoArchive "train2017.zip") 19336861798 8
Download-File "http://images.cocodataset.org/zips/val2017.zip" (Join-Path $cocoArchive "val2017.zip") 815585330
Download-File "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" (Join-Path $cocoArchive "annotations_trainval2017.zip") 252907541
Expand-ZipIfNeeded (Join-Path $cocoArchive "train2017.zip") $cocoRoot (Join-Path $cocoRoot "train2017")
Expand-ZipIfNeeded (Join-Path $cocoArchive "val2017.zip") $cocoRoot (Join-Path $cocoRoot "val2017")
Expand-ZipIfNeeded (Join-Path $cocoArchive "annotations_trainval2017.zip") $cocoRoot (Join-Path $cocoRoot "annotations\instances_train2017.json")

# LVIS v1 annotations for OV-LVIS. Images are COCO train2017/val2017.
$lvisArchive = Join-Path $ArchiveRoot "lvis"
$lvisAnn = Join-Path $Root "lvis\annotations"
Ensure-Dir $lvisArchive
Ensure-Dir $lvisAnn
Download-File "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip" (Join-Path $lvisArchive "lvis_v1_train.json.zip") 350264821
Download-File "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip" (Join-Path $lvisArchive "lvis_v1_val.json.zip") 64026968
Expand-ZipIfNeeded (Join-Path $lvisArchive "lvis_v1_train.json.zip") $lvisAnn (Join-Path $lvisAnn "lvis_v1_train.json")
Expand-ZipIfNeeded (Join-Path $lvisArchive "lvis_v1_val.json.zip") $lvisAnn (Join-Path $lvisAnn "lvis_v1_val.json")

# PASCAL VOC 2007 test set for supplemental transfer evaluation.
$vocArchive = Join-Path $ArchiveRoot "voc2007"
$vocRoot = Join-Path $Root "voc2007"
Ensure-Dir $vocArchive
Ensure-Dir $vocRoot
Download-File "https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar" (Join-Path $vocArchive "VOCtest_06-Nov-2007.tar") 451020800
Expand-TarIfNeeded (Join-Path $vocArchive "VOCtest_06-Nov-2007.tar") $vocRoot (Join-Path $vocRoot "VOCdevkit\VOC2007\ImageSets\Main\test.txt")

if (-not $SkipObjects365) {
    # Objects365 v2 validation set for supplemental transfer evaluation only.
    $objArchive = Join-Path $ArchiveRoot "objects365v2_val"
    $objVal = Join-Path $Root "objects365v2\val"
    Ensure-Dir $objArchive
    Ensure-Dir $objVal
    $objBase = "https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86"
    Download-File "$objBase/val/zhiyuan_objv2_val.json" (Join-Path $objVal "zhiyuan_objv2_val.json") 269422006
    for ($i = 0; $i -lt 44; $i++) {
        if ($i -lt 16) {
            $url = "$objBase/val/images/v1/patch$i.tar.gz"
        } else {
            $url = "$objBase/val/images/v2/patch$i.tar.gz"
        }
        $archive = Join-Path $objArchive "patch$i.tar.gz"
        Download-File $url $archive
        $marker = Join-Path $objVal ".patch$i.extracted"
        if (-not (Test-Path -LiteralPath $marker)) {
            Write-Host "EXTRACT $archive -> $objVal"
            tar.exe -xzf $archive -C $objVal
            New-Item -ItemType File -Force -Path $marker | Out-Null
        } else {
            Write-Host "SKIP extracted Objects365 patch$i"
        }
    }
}

Write-Host "Dataset download/extract stage complete: $Root"
